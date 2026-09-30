#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the Dex3-1 hand / Dex1-1 gripper DDS driver, a Dex3 robot and the ZMQ SDK stand-in's
hand topics. Meant to be run in an environment where the Unitree SDK is installed."""

from __future__ import annotations

import contextlib
import copy
import time
from collections.abc import Callable
from unittest.mock import MagicMock, patch

import pytest

from lerobot.utils.import_utils import _unitree_sdk_available

if not _unitree_sdk_available:
    pytest.skip("Unitree SDK not available", allow_module_level=True)

from unitree_sdk2py.idl.default import (
    unitree_go_msg_dds__MotorCmd_,
    unitree_go_msg_dds__MotorState_,
    unitree_hg_msg_dds__HandCmd_,
    unitree_hg_msg_dds__HandState_,
)
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_

from lerobot.robots.unitree_g1 import unitree_sdk2_socket as sock
from lerobot.robots.unitree_g1.config_unitree_g1 import UnitreeG1Config
from lerobot.robots.unitree_g1.dex_hands import (
    DEX1_REAL_CLOSED_Q,
    DEX1_REAL_OPEN_Q,
    DEX_GAINS,
    DexHandDriver,
    dex_topics,
)
from lerobot.robots.unitree_g1.end_effectors import DEX1, DEX3, DEX3_JOINTS, HAND_CLOSURE_KEYS, HAND_SIDES
from lerobot.robots.unitree_g1.g1_utils import BODY_KEYS

# Motor order of the real Dex3 right hand (the sim and the real left hand use DEX3_JOINTS order).
REAL_RIGHT_DEX3_ORDER = ("Thumb0", "Thumb1", "Thumb2", "Index0", "Index1", "Middle0", "Middle1")


class FakeChannels:
    """Fake SDK channel classes recording every publisher/subscriber by topic.

    Subscribers `Init`-ed with a handler immediately deliver `state_factory(topic, msg_type)` (unless it
    returns None), like the first DDS sample arriving on connect.
    """

    def __init__(self, state_factory: Callable | None = None, lowstate=None):
        self.publishers: dict[str, FakePublisher] = {}
        self.subscribers: dict[str, FakeSubscriber] = {}
        self.state_factory = state_factory
        self.lowstate = lowstate
        channels = self

        class Publisher(FakePublisher):
            def __init__(self, topic, msg_type):
                super().__init__(topic, msg_type)
                channels.publishers[topic] = self

        class Subscriber(FakeSubscriber):
            def __init__(self, topic, msg_type):
                super().__init__(topic, msg_type, channels)
                channels.subscribers[topic] = self

        self.Publisher = Publisher
        self.Subscriber = Subscriber


class FakePublisher:
    def __init__(self, topic, msg_type):
        self.topic = topic
        self.msg_type = msg_type
        self.initialized = False
        self.written: list = []

    def Init(self):  # noqa: N802
        self.initialized = True

    def Write(self, msg):  # noqa: N802
        self.written.append(copy.deepcopy(msg))


class FakeSubscriber:
    def __init__(self, topic, msg_type, channels: FakeChannels):
        self.topic = topic
        self.msg_type = msg_type
        self.channels = channels
        self.handler = None
        self.queue_len = None
        self.closed = False

    def Init(self, handler=None, queueLen=0):  # noqa: N802, N803
        self.handler = handler
        self.queue_len = queueLen
        if handler is not None and self.channels.state_factory is not None:
            msg = self.channels.state_factory(self.topic, self.msg_type)
            if msg is not None:
                handler(msg)

    def Read(self):  # noqa: N802
        return self.channels.lowstate

    def Close(self):  # noqa: N802
        self.closed = True

    def deliver(self, msg):
        self.handler(msg)


def _hand_state(q: list[float]) -> HandState_:
    msg = unitree_hg_msg_dds__HandState_()
    for motor, value in zip(msg.motor_state, q, strict=False):
        motor.q = value
    return msg


def _gripper_state(q: float) -> MotorStates_:
    state = unitree_go_msg_dds__MotorState_()
    state.q = q
    return MotorStates_(states=[state])


def _default_state(topic, msg_type):
    """A state message of the right SDK type for `topic`, all joints at 0."""
    return _gripper_state(DEX1_REAL_OPEN_Q) if msg_type is MotorStates_ else _hand_state([0.0] * 7)


def _connected_driver(spec, is_simulation, state_factory=_default_state):
    channels = FakeChannels(state_factory)
    driver = DexHandDriver(spec, is_simulation, channels.Publisher, channels.Subscriber)
    driver.connect(timeout_s=0.5)
    return driver, channels


def _last_cmd_motors(channels, topic):
    msg = channels.publishers[topic].written[-1]
    return msg.motor_cmd if hasattr(msg, "motor_cmd") else msg.cmds


# ---------------------------------------------------------------------------
# DexHandDriver
# ---------------------------------------------------------------------------


class TestDexTopics:
    @pytest.mark.parametrize(
        "end_effector, is_simulation, family",
        [("dex3", True, "dex3"), ("dex3", False, "dex3"), ("dex1", True, "dex3"), ("dex1", False, "dex1")],
    )
    @pytest.mark.parametrize("side", ["left", "right"])
    def test_topics(self, end_effector, is_simulation, family, side):
        assert dex_topics(end_effector, is_simulation, side) == (
            f"rt/{family}/{side}/cmd",
            f"rt/{family}/{side}/state",
        )

    @pytest.mark.parametrize(
        "spec, is_simulation, cmd_type, state_type",
        [
            (DEX3, True, HandCmd_, HandState_),
            (DEX3, False, HandCmd_, HandState_),
            (DEX1, True, HandCmd_, HandState_),
            (DEX1, False, MotorCmds_, MotorStates_),
        ],
        ids=["dex3-sim", "dex3-real", "dex1-sim", "dex1-real"],
    )
    def test_connect_creates_channels_per_side(self, spec, is_simulation, cmd_type, state_type):
        driver, channels = _connected_driver(spec, is_simulation)
        for side in HAND_SIDES:
            cmd_topic, state_topic = dex_topics(spec.name, is_simulation, side)
            publisher = channels.publishers[cmd_topic]
            subscriber = channels.subscribers[state_topic]
            assert publisher.msg_type is cmd_type
            assert publisher.initialized
            assert subscriber.msg_type is state_type
            assert subscriber.handler is not None
            assert subscriber.queue_len == 1
        assert len(channels.publishers) == len(channels.subscribers) == 2
        driver.disconnect()
        assert all(subscriber.closed for subscriber in channels.subscribers.values())


class TestDexHandDriverWrite:
    @pytest.mark.parametrize("side", ["left", "right"])
    def test_sim_dex1_writes_both_finger_slots(self, side):
        driver, channels = _connected_driver(DEX1, is_simulation=True)
        driver.write({f"k{side.capitalize()}Gripper.q": 0.01})
        motors = _last_cmd_motors(channels, f"rt/dex3/{side}/cmd")
        assert motors[0].q == pytest.approx(0.01)
        assert motors[1].q == pytest.approx(0.01)
        assert all(motor.q == 0.0 for motor in motors[2:])
        other = "left" if side == "right" else "right"
        assert channels.publishers[f"rt/dex3/{other}/cmd"].written == []

    @pytest.mark.parametrize(
        "q, motor_q",
        [
            (DEX1.open_q["left"][0], DEX1_REAL_OPEN_Q),
            (DEX1.closed_q["left"][0], DEX1_REAL_CLOSED_Q),
            (
                (DEX1.open_q["left"][0] + DEX1.closed_q["left"][0]) / 2,
                (DEX1_REAL_OPEN_Q + DEX1_REAL_CLOSED_Q) / 2,
            ),
        ],
    )
    def test_real_dex1_converts_to_motor_units(self, q, motor_q):
        driver, channels = _connected_driver(DEX1, is_simulation=False)
        driver.write({"kLeftGripper.q": q, "kRightGripper.q": q})
        for side in HAND_SIDES:
            msg = channels.publishers[f"rt/dex1/{side}/cmd"].written[-1]
            assert isinstance(msg, MotorCmds_)
            assert len(msg.cmds) == 1
            assert msg.cmds[0].q == pytest.approx(motor_q)

    def test_real_dex3_right_hand_reorders_slots(self):
        driver, channels = _connected_driver(DEX3, is_simulation=False)
        targets = {key: 0.1 * (i + 1) for i, key in enumerate(DEX3.joint_keys())}
        driver.write(targets)
        right = _last_cmd_motors(channels, "rt/dex3/right/cmd")
        for joint in DEX3_JOINTS:
            slot = REAL_RIGHT_DEX3_ORDER.index(joint)
            assert right[slot].q == pytest.approx(targets[f"kRightHand{joint}.q"])
        assert right[3].q == pytest.approx(targets["kRightHandIndex0.q"])
        left = _last_cmd_motors(channels, "rt/dex3/left/cmd")
        for slot, key in enumerate(DEX3.joint_keys("left")):
            assert left[slot].q == pytest.approx(targets[key])

    def test_sim_dex3_right_hand_keeps_model_order(self):
        driver, channels = _connected_driver(DEX3, is_simulation=True)
        targets = {key: 0.1 * (i + 1) for i, key in enumerate(DEX3.joint_keys("right"))}
        driver.write(targets)
        right = _last_cmd_motors(channels, "rt/dex3/right/cmd")
        assert [motor.q for motor in right] == pytest.approx(list(targets.values()))
        assert channels.publishers["rt/dex3/left/cmd"].written == []

    @pytest.mark.parametrize(("end_effector", "is_simulation"), list(DEX_GAINS))
    def test_gains_per_end_effector_and_mode(self, end_effector, is_simulation):
        spec = {"dex1": DEX1, "dex3": DEX3}[end_effector]
        driver, channels = _connected_driver(spec, is_simulation)
        kp, kd = DEX_GAINS[(end_effector, is_simulation)]
        assert (driver.kp, driver.kd) == (kp, kd)
        driver.write(dict.fromkeys(spec.joint_keys("left"), 0.0))
        cmd_topic, _ = dex_topics(end_effector, is_simulation, "left")
        written = [m for m in _last_cmd_motors(channels, cmd_topic) if m.kp != 0.0]
        assert written
        assert all((m.kp, m.kd, m.dq, m.tau) == (kp, kd, 0.0, 0.0) for m in written)

    def test_real_and_sim_dex1_gains_differ(self):
        assert DEX_GAINS[("dex1", True)] != DEX_GAINS[("dex1", False)]

    def test_dex3_mode_byte(self):
        driver, channels = _connected_driver(DEX3, is_simulation=False)
        driver.write(dict.fromkeys(DEX3.joint_keys(), 0.0))
        for side in HAND_SIDES:
            for slot, motor in enumerate(_last_cmd_motors(channels, f"rt/dex3/{side}/cmd")):
                assert motor.mode == (slot & 0x0F) | 0x10

    def test_dex1_leaves_mode_alone(self):
        driver, channels = _connected_driver(DEX1, is_simulation=True)
        driver.write({"kLeftGripper.q": 0.0})
        assert all(motor.mode == 0 for motor in _last_cmd_motors(channels, "rt/dex3/left/cmd"))

    def test_partial_targets_only_update_given_joints(self):
        driver, channels = _connected_driver(DEX3, is_simulation=True)
        driver.write({"kLeftHandThumb1.q": 0.3})
        driver.write({"kLeftHandMiddle0.q": -0.5})
        left = _last_cmd_motors(channels, "rt/dex3/left/cmd")
        assert left[1].q == pytest.approx(0.3)
        assert left[3].q == pytest.approx(-0.5)
        assert left[0].q == 0.0


class TestDexHandDriverRead:
    def test_read_via_state_handler_sim_dex3(self):
        driver, channels = _connected_driver(DEX3, is_simulation=True)
        values = [0.1 * (i + 1) for i in range(7)]
        channels.subscribers["rt/dex3/left/state"].deliver(_hand_state(values))
        q = driver.read()
        assert set(q) == set(DEX3.joint_keys())
        assert [q[key] for key in DEX3.joint_keys("left")] == pytest.approx(values)

    def test_read_real_dex3_right_hand_reorders_slots(self):
        driver, channels = _connected_driver(DEX3, is_simulation=False)
        real = [0.1 * (i + 1) for i in range(7)]  # real motor order: thumb, index, middle
        channels.subscribers["rt/dex3/right/state"].deliver(_hand_state(real))
        q = driver.read()
        for joint in DEX3_JOINTS:
            assert q[f"kRightHand{joint}.q"] == pytest.approx(real[REAL_RIGHT_DEX3_ORDER.index(joint)])

    @pytest.mark.parametrize(
        "motor_q, q",
        [(DEX1_REAL_OPEN_Q, DEX1.open_q["left"][0]), (DEX1_REAL_CLOSED_Q, DEX1.closed_q["left"][0])],
    )
    def test_read_real_dex1_converts_from_motor_units(self, motor_q, q):
        driver, channels = _connected_driver(DEX1, is_simulation=False)
        channels.subscribers["rt/dex1/right/state"].deliver(_gripper_state(motor_q))
        assert driver.read()["kRightGripper.q"] == pytest.approx(q)

    def test_real_dex1_unit_conversion_round_trip(self):
        driver, _ = _connected_driver(DEX1, is_simulation=False)
        for q in (-0.02, 0.0, 0.01, 0.0245):
            assert driver._from_motor(driver._to_motor(q)) == pytest.approx(q)

    def test_read_sim_dex1_uses_first_finger_slot(self):
        driver, channels = _connected_driver(DEX1, is_simulation=True)
        channels.subscribers["rt/dex3/left/state"].deliver(_hand_state([0.01, 0.02]))
        assert driver.read()["kLeftGripper.q"] == pytest.approx(0.01)

    def test_read_returns_a_copy(self):
        driver, _ = _connected_driver(DEX3, is_simulation=True)
        q = driver.read()
        q["kLeftHandThumb0.q"] = 99.0
        assert driver.read()["kLeftHandThumb0.q"] == 0.0

    def test_connect_times_out_without_state(self):
        channels = FakeChannels(
            state_factory=lambda topic, msg_type: None
            if "right" in topic
            else _default_state(topic, msg_type)
        )
        driver = DexHandDriver(DEX3, True, channels.Publisher, channels.Subscriber)
        with pytest.raises(TimeoutError, match="kRightHandThumb0.q"):
            driver.connect(timeout_s=0.05)


# ---------------------------------------------------------------------------
# UnitreeG1 with a Dex3 hand
# ---------------------------------------------------------------------------


def _make_lowstate_msg_mock():
    msg = MagicMock()
    msg.motor_state.__getitem__ = lambda self, idx, _motors={}: _motors.setdefault(
        idx, MagicMock(q=idx * 0.1, dq=idx * 0.01, tau_est=idx * 0.001, temperature=30.0 + idx)
    )
    msg.imu_state.quaternion = [1.0, 0.0, 0.0, 0.0]
    msg.imu_state.gyroscope = [0.1, 0.2, 0.3]
    msg.imu_state.accelerometer = [0.0, 0.0, 9.81]
    msg.imu_state.rpy = [0.0, 0.0, 0.0]
    msg.imu_state.temperature = 25.0
    msg.wireless_remote = b"\x00" * 40
    msg.mode_machine = 0
    return msg


@contextlib.contextmanager
def _mocked_dex_robot(end_effector="dex3", state_factory=_default_state, **config_kwargs):
    """A connected simulated 29dof UnitreeG1 with a Dex hand, on fake SDK channels."""
    channels = FakeChannels(state_factory, lowstate=_make_lowstate_msg_mock())
    lowcmd = MagicMock()
    lowcmd.motor_cmd = [MagicMock() for _ in range(35)]
    fake_inner_env = MagicMock()
    fake_inner_env.simulator = None
    fake_env_wrapper = {"hub_env": {0: MagicMock(envs=[fake_inner_env])}}
    module = "lerobot.robots.unitree_g1.unitree_g1"
    with (
        patch(f"{module}.make_cameras_from_configs", return_value={}),
        patch(f"{module}._SDKChannelFactoryInitialize", MagicMock()),
        patch(f"{module}._SDKChannelPublisher", channels.Publisher),
        patch(f"{module}._SDKChannelSubscriber", channels.Subscriber),
        patch(f"{module}.unitree_hg_msg_dds__LowCmd_", MagicMock(return_value=lowcmd)),
        patch(f"{module}.CRC", MagicMock(return_value=MagicMock(Crc=MagicMock(return_value=0)))),
        patch("lerobot.envs.make_env", return_value=fake_env_wrapper),
    ):
        from lerobot.robots.unitree_g1.unitree_g1 import UnitreeG1

        cfg = UnitreeG1Config(body="29dof", end_effector=end_effector, is_simulation=True, **config_kwargs)
        robot = UnitreeG1(cfg)
        try:
            yield robot, channels
        finally:
            if robot.is_connected:
                robot.disconnect()


class TestDex3Robot:
    def test_embodiment_wiring(self):
        with _mocked_dex_robot() as (robot, _):
            assert isinstance(robot.dex_hand, DexHandDriver)
            assert robot.dex_hand.spec is DEX3
            assert robot.headhand is None
            assert robot.name == "unitree_g1_29dof_dex3"
            assert list(robot.action_features) == [*BODY_KEYS, *HAND_CLOSURE_KEYS]
            assert robot.is_calibrated
            robot.calibrate()  # no head/hand bridge: nothing to calibrate

    def test_send_action_closure_writes_closed_pose(self):
        with _mocked_dex_robot() as (robot, channels):
            robot.connect(calibrate=False)
            sent = robot.send_action({"kLeftHand.closure": 1.0})
            assert sent["kLeftHand.closure"] == 1.0
            assert [sent[key] for key in DEX3.joint_keys("left")] == pytest.approx(DEX3.closed_q["left"])
            left = _last_cmd_motors(channels, "rt/dex3/left/cmd")
            assert [motor.q for motor in left] == pytest.approx(DEX3.closed_q["left"])
            assert channels.publishers["rt/dex3/right/cmd"].written == []

    def test_send_action_clips_closure(self):
        with _mocked_dex_robot() as (robot, channels):
            robot.connect(calibrate=False)
            sent = robot.send_action({"kRightHand.closure": 1.5})
            assert sent["kRightHand.closure"] == 1.0
            right = _last_cmd_motors(channels, "rt/dex3/right/cmd")
            assert [motor.q for motor in right] == pytest.approx(DEX3.closed_q["right"])

    def test_get_observation_adds_closure_from_hand_state(self):
        def state(topic, msg_type):
            side = "left" if "left" in topic else "right"
            return _hand_state(list(DEX3.closure_to_q(side, 0.4 if side == "left" else 0.8)))

        with _mocked_dex_robot(state_factory=state) as (robot, _):
            robot.connect(calibrate=False)
            obs = robot.get_observation()
            assert obs["kLeftHand.closure"] == pytest.approx(0.4)
            assert obs["kRightHand.closure"] == pytest.approx(0.8)
            assert all(key in obs for key in DEX3.joint_keys())

    def test_per_motor_send_action_forwards_joints(self):
        with _mocked_dex_robot(hand_representation="per_motor") as (robot, channels):
            robot.connect(calibrate=False)
            assert list(robot.action_features) == [*BODY_KEYS, *DEX3.joint_keys()]
            targets = dict(zip(DEX3.joint_keys("right"), DEX3.closure_to_q("right", 0.5), strict=True))
            sent = robot.send_action({**targets, "kRightHand.closure": 1.0})
            # The closure key is not interpreted: the joints are sent as given.
            assert [sent[key] for key in targets] == pytest.approx(list(targets.values()))
            right = _last_cmd_motors(channels, "rt/dex3/right/cmd")
            assert [motor.q for motor in right] == pytest.approx(list(targets.values()))
            assert "kRightHand.closure" not in robot.get_observation()

    def test_dex1_robot_uses_sim_dex3_topics(self):
        with _mocked_dex_robot(end_effector="dex1") as (robot, channels):
            robot.connect(calibrate=False)
            robot.send_action({"kLeftHand.closure": 0.0})
            left = _last_cmd_motors(channels, "rt/dex3/left/cmd")
            assert left[0].q == left[1].q == pytest.approx(DEX1.open_q["left"][0])

    def test_connect_without_hand_state_times_out_and_disconnects(self):
        with (
            _mocked_dex_robot(state_factory=lambda topic, msg_type: None) as (robot, _),
            patch(
                "lerobot.robots.unitree_g1.dex_hands.DexHandDriver.connect",
                side_effect=TimeoutError("no state"),
            ),
        ):
            with pytest.raises(TimeoutError):
                robot.connect(calibrate=False)
            assert not robot.is_connected


# ---------------------------------------------------------------------------
# unitree_sdk2_socket (ZMQ stand-in for the SDK on real hardware)
# ---------------------------------------------------------------------------


class TestSdk2Socket:
    def test_topic_ports_are_unique(self):
        ports = list(sock.TOPIC_PORTS.values())
        assert len(ports) == len(set(ports))
        # 6002/6003 belong to the head/hand bridge.
        assert not {6002, 6003} & set(ports)

    def test_every_dex_topic_is_bridged(self):
        for end_effector in ("dex1", "dex3"):
            for side in HAND_SIDES:
                for topic in dex_topics(end_effector, False, side):
                    assert sock.topic_port(topic) == sock.TOPIC_PORTS[topic]
        assert sock.topic_port("rt/lowcmd") == sock.LOWCMD_PORT
        assert sock.topic_port("rt/lowstate") == sock.LOWSTATE_PORT

    def test_topic_port_raises_on_unknown_topic(self):
        with pytest.raises(ValueError, match="not bridged"):
            sock.topic_port("rt/not_a_topic")

    def test_lowcmd_to_dict_hand_cmd(self):
        msg = unitree_hg_msg_dds__HandCmd_()
        msg.motor_cmd[2].q = 0.5
        msg.motor_cmd[2].kp = 1.5
        msg.motor_cmd[2].mode = 0x12
        out = sock.lowcmd_to_dict("rt/dex3/left/cmd", msg)
        assert out["topic"] == "rt/dex3/left/cmd"
        assert len(out["data"]["motor_cmd"]) == 7
        assert out["data"]["motor_cmd"][2] == {
            "mode": 0x12,
            "q": 0.5,
            "dq": 0.0,
            "kp": 1.5,
            "kd": 0.0,
            "tau": 0.0,
        }
        assert "mode_pr" not in out["data"]

    def test_lowcmd_to_dict_motor_cmds(self):
        cmd = unitree_go_msg_dds__MotorCmd_()
        cmd.q = 5.4
        cmd.kd = 0.05
        out = sock.lowcmd_to_dict("rt/dex1/right/cmd", MotorCmds_(cmds=[cmd]))
        assert out["data"]["motor_cmd"] == [
            {"mode": 0, "q": 5.4, "dq": 0.0, "kp": 0.0, "kd": 0.05, "tau": 0.0}
        ]

    def test_motor_states_msg_exposes_both_attributes(self):
        msg = sock.MotorStatesMsg({"motor_state": [{"q": 0.1}, {"q": 0.2, "dq": 1.0}]})
        assert [m.q for m in msg.motor_state] == [0.1, 0.2]
        assert msg.states is msg.motor_state
        assert msg.states[1].dq == 1.0
        assert sock.MotorStatesMsg({}).states == []

    def test_driver_reads_motor_states_msg(self):
        driver, channels = _connected_driver(DEX1, is_simulation=False)
        channels.subscribers["rt/dex1/left/state"].deliver(
            sock.MotorStatesMsg({"motor_state": [{"q": DEX1_REAL_CLOSED_Q}]})
        )
        assert driver.read()["kLeftGripper.q"] == pytest.approx(DEX1.closed_q["left"][0])

    def test_publisher_write_before_init_raises(self):
        publisher = sock.ChannelPublisher("rt/dex3/left/cmd", HandCmd_)
        with pytest.raises(RuntimeError):
            publisher.Write(unitree_hg_msg_dds__HandCmd_())

    def test_hand_topics_round_trip_over_zmq(self):
        zmq = pytest.importorskip("zmq")
        ctx = zmq.Context.instance()
        cmd_pull = ctx.socket(zmq.PULL)
        state_pub = ctx.socket(zmq.PUB)
        with patch.dict(sock.TOPIC_PORTS, {}, clear=False):
            sock.TOPIC_PORTS["rt/dex3/left/cmd"] = cmd_pull.bind_to_random_port("tcp://127.0.0.1")
            sock.TOPIC_PORTS["rt/dex3/left/state"] = state_pub.bind_to_random_port("tcp://127.0.0.1")
            sock.ChannelFactoryInitialize(0, UnitreeG1Config(robot_ip="127.0.0.1"))
            received = []
            subscriber = sock.ChannelSubscriber("rt/dex3/left/state", HandState_)
            subscriber.Init(received.append, 1)
            publisher = sock.ChannelPublisher("rt/dex3/left/cmd", HandCmd_)
            publisher.Init()
            try:
                cmd = unitree_hg_msg_dds__HandCmd_()
                cmd.motor_cmd[0].q = 0.25
                publisher.Write(cmd)
                assert cmd_pull.poll(2000)
                assert sock.json.loads(cmd_pull.recv())["data"]["motor_cmd"][0]["q"] == 0.25

                deadline = time.time() + 2.0
                while not received and time.time() < deadline:
                    state_pub.send(sock.json.dumps({"data": {"motor_state": [{"q": 0.7}]}}).encode())
                    time.sleep(0.02)
                assert received
                assert isinstance(received[0], sock.MotorStatesMsg)
                assert received[0].motor_state[0].q == 0.7
            finally:
                subscriber.Close()
                publisher._sock.close(linger=0)
                cmd_pull.close(linger=0)
                state_pub.close(linger=0)
