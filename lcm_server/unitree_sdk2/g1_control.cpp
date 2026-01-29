#include <yaml-cpp/yaml.h>

#include <cmath>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <lcm/lcm-cpp.hpp>
#include <thread>
#include <chrono>

// DDS
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

// IDL
#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/idl/go2/SportModeState_.hpp>

// LCM
#include "low_cmd_lcmt.hpp"
#include "low_state_lcmt.hpp"
#include "sport_state_lcmt.hpp"
// #include "pd_tau_targets_lcmt.hpp"
// #include "state_estimator_lcmt.hpp"
// #include "body_control_data_lcmt.hpp"
// #include "rc_command_lcmt.hpp"
// #include "arm_action_lcmt.hpp"
// gamepad
#include "unitree/common/thread/thread.hpp"
// #include "unitree/idl/go2/WirelessController_.hpp"
// #include "/home/unitree/unitree_sdk2/example/wireless_controller/advanced_gamepad.hpp"

// #define TOPIC_JOYSTICK "rt/wirelesscontroller"

static const std::string HG_CMD_TOPIC = "rt/lowcmd";
static const std::string HG_STATE_TOPIC = "rt/lowstate";
static const std::string GO_SPORT_STATE_TOPIC = "rt/odommodestate";

static const std::string LCM_CMD_TOPIC = "low_cmd_topic";
static const std::string LCM_STATE_TOPIC = "low_state_topic";
static const std::string LCM_SPORT_STATE_TOPIC = "sport_state_topic";

using namespace unitree::common;
using namespace unitree::robot;
using namespace unitree_hg::msg::dds_;

const int G1_NUM_MOTOR = 29; // not included 7*2 of two hands

typedef union {
  struct {
    uint8_t R1 : 1;
    uint8_t L1 : 1;
    uint8_t start : 1;
    uint8_t select : 1;
    uint8_t R2 : 1;
    uint8_t L2 : 1;
    uint8_t F1 : 1;
    uint8_t F2 : 1;
    uint8_t A : 1;
    uint8_t B : 1;
    uint8_t X : 1;
    uint8_t Y : 1;
    uint8_t up : 1;
    uint8_t right : 1;
    uint8_t down : 1;
    uint8_t left : 1;
  } components;
  uint16_t value;
} xKeySwitchUnion;

typedef struct {
  uint8_t head[2];
  xKeySwitchUnion btn;
  float lx;
  float rx;
  float ry;
  float L2;
  float ly;

  uint8_t idle[16];
} xRockerBtnDataStruct;

template <typename T>
class DataBuffer {
 public:
  void SetData(const T &newData) {
    std::unique_lock<std::shared_mutex> lock(mutex);
    data = std::make_shared<T>(newData);
  }

  std::shared_ptr<const T> GetData() {
    std::shared_lock<std::shared_mutex> lock(mutex);
    return data ? data : nullptr;
  }

  void Clear() {
    std::unique_lock<std::shared_mutex> lock(mutex);
    data = nullptr;
  }

 private:
  std::shared_ptr<T> data;
  std::shared_mutex mutex;
};

struct ImuState {
  std::array<float, 3> rpy = {};
  std::array<float, 3> gyroscope = {};
  std::array<float, 4> quaternion = {};
  std::array<float, 3> accelerometer = {};
};

struct RCState{
  std::array<uint8_t, 40> wireless_remote = {};
};

struct MotorCommand {
  std::array<float, G1_NUM_MOTOR> q_target = {};
  std::array<float, G1_NUM_MOTOR> dq_target = {};
  std::array<float, G1_NUM_MOTOR> kp = {};
  std::array<float, G1_NUM_MOTOR> kd = {};
  std::array<float, G1_NUM_MOTOR> tau_ff = {};
};

struct MotorState {
  std::array<float, G1_NUM_MOTOR> q = {};
  std::array<float, G1_NUM_MOTOR> dq = {};
  std::array<float, G1_NUM_MOTOR> ddq = {};
  std::array<float, G1_NUM_MOTOR> tau_est = {};
};

struct SportState {
  int64_t error_code = {};
  uint8_t mode = {};
  float progress = {};
  uint8_t gait_type = {};
  float foot_raise_height = {};
  std::array<float, 3> position = {};
  float body_height = {};
  std::array<float, 3> velocity = {};
  float yaw_speed = {};
};

enum PRorAB { PR = 0, AB = 1 };

enum G1JointIndex {
  LeftHipPitch = 0,
  LeftHipRoll = 1,
  LeftHipYaw = 2,
  LeftKnee = 3,
  LeftAnklePitch = 4,
  LeftAnkleB = 4,
  LeftAnkleRoll = 5,
  LeftAnkleA = 5,
  RightHipPitch = 6,
  RightHipRoll = 7,
  RightHipYaw = 8,
  RightKnee = 9,
  RightAnklePitch = 10,
  RightAnkleB = 10,
  RightAnkleRoll = 11,
  RightAnkleA = 11,
  WaistYaw = 12,
  WaistRoll = 13,        // NOTE INVALID for g1 23dof/29dof with waist locked
  WaistA = 13,           // NOTE INVALID for g1 23dof/29dof with waist locked
  WaistPitch = 14,       // NOTE INVALID for g1 23dof/29dof with waist locked
  WaistB = 14,           // NOTE INVALID for g1 23dof/29dof with waist locked
  LeftShoulderPitch = 15,
  LeftShoulderRoll = 16,
  LeftShoulderYaw = 17,
  LeftElbow = 18,
  LeftWristRoll = 19,
  LeftWristPitch = 20,   // NOTE INVALID for g1 23dof
  LeftWristYaw = 21,     // NOTE INVALID for g1 23dof
  RightShoulderPitch = 22,
  RightShoulderRoll = 23,
  RightShoulderYaw = 24,
  RightElbow = 25,
  RightWristRoll = 26,
  RightWristPitch = 27,  // NOTE INVALID for g1 23dof
  RightWristYaw = 28     // NOTE INVALID for g1 23dof
};

inline uint32_t Crc32Core(uint32_t *ptr, uint32_t len) {
  uint32_t xbit = 0;
  uint32_t data = 0;
  uint32_t CRC32 = 0xFFFFFFFF;
  const uint32_t dwPolynomial = 0x04c11db7;
  for (uint32_t i = 0; i < len; i++) {
    xbit = 1 << 31;
    data = ptr[i];
    for (uint32_t bits = 0; bits < 32; bits++) {
      if (CRC32 & 0x80000000) {
        CRC32 <<= 1;
        CRC32 ^= dwPolynomial;
      } else
        CRC32 <<= 1;
      if (data & xbit) CRC32 ^= dwPolynomial;

      xbit >>= 1;
    }
  }
  return CRC32;
};

class G1Control {
 private:
  double time_;
  double control_dt_;  // [2ms] thus 500hz
  PRorAB mode_;
  uint8_t mode_machine_;
  std::vector<std::vector<double>> frames_data_;

  DataBuffer<MotorState> motor_state_buffer_;
  DataBuffer<MotorCommand> motor_command_buffer_;
  DataBuffer<ImuState> imu_state_buffer_;
  DataBuffer<RCState> rc_state_buffer_;
  DataBuffer<SportState> sport_state_buffer_;


  ChannelPublisherPtr<unitree_hg::msg::dds_::LowCmd_> lowcmd_publisher_;
  ChannelSubscriberPtr<unitree_hg::msg::dds_::LowState_> lowstate_subscriber_;
  ChannelSubscriberPtr<unitree_go::msg::dds_::SportModeState_> sportstate_subscriber_;
  // ThreadPtr command_writer_ptr_, control_thread_ptr_, joystick_thread_ptr_;
  ThreadPtr command_writer_ptr_, control_thread_ptr_;
  
  std::string lcm_url="udpm://239.255.76.68:7667?ttl=255";
  lcm::LCM _simpleLCM = lcm::LCM(lcm_url);
  // lcm::LCM _simpleLCM;
  std::thread _simple_LCM_thread;
  // bool _firstRun;
  bool _firstCommandReceived;
  
  // init lcm data
  sport_state_lcmt lcm_sport_state = {0};
  low_state_lcmt lcm_low_state = {0};
  low_cmd_lcmt lcm_low_cmd = {0};

  // state_estimator_lcmt body_state_simple = {0};
  // body_control_data_lcmt joint_state_simple = {0};
  // pd_tau_targets_lcmt joint_command_simple = {0};
  // arm_action_lcmt arm_action_simple = {0};
  // rc_command_lcmt rc_command = {0};
  // Gamepad gamepad;
  // unitree_go::msg::dds_::WirelessController_ joystick_msg;
  // ChannelSubscriberPtr<unitree_go::msg::dds_::WirelessController_> joystick_subscriber;
  // std::mutex joystick_mutex;

 public:
  G1Control(std::string networkInterface)
      : time_(0.0),
        control_dt_(0.002),
        mode_(PR),
        mode_machine_(0) {
    ChannelFactory::Instance()->Init(0, networkInterface);

    // create dds publisher
    lowcmd_publisher_.reset(
        new ChannelPublisher<unitree_hg::msg::dds_::LowCmd_>(HG_CMD_TOPIC));
    lowcmd_publisher_->InitChannel();

    // create dds subscriber
    lowstate_subscriber_.reset(
        new ChannelSubscriber<unitree_hg::msg::dds_::LowState_>(
            HG_STATE_TOPIC));
    lowstate_subscriber_->InitChannel(
        std::bind(&G1Control::LowStateHandler, this, std::placeholders::_1), 1);

    sportstate_subscriber_.reset(
        new ChannelSubscriber<unitree_go::msg::dds_::SportModeState_>(
            GO_SPORT_STATE_TOPIC));
    sportstate_subscriber_->InitChannel(
        std::bind(&G1Control::SportStateHandler, this, std::placeholders::_1), 1);
    SportState empyt_ss;
    sport_state_buffer_.SetData(empyt_ss);
    // create threads
    command_writer_ptr_ =CreateRecurrentThreadEx("command_writer", UT_CPU_ID_NONE, 2000, &G1Control::LowCommandWriter, this);  // interval 2ms, 500Hz
    control_thread_ptr_ = CreateRecurrentThreadEx("control", UT_CPU_ID_NONE, 2000, &G1Control::Control, this);  // interval 2ms, 500Hz

    // add lcm subscriber
    _simpleLCM.subscribe(LCM_CMD_TOPIC, &G1Control::handleActionLCM, this);
    _simple_LCM_thread = std::thread(&G1Control::_simpleLCMThread, this);
    _firstCommandReceived = false;
  }

  void _simpleLCMThread(){
    while(true){
        _simpleLCM.handle();
    }
  }

  void handleActionLCM(const lcm::ReceiveBuffer *rbuf, const std::string & chan, const low_cmd_lcmt * msg){
    (void) rbuf;
    (void) chan;
    lcm_low_cmd = *msg;
    if (_firstCommandReceived == false){
      _firstCommandReceived = true;
      std::cout << "First command received" << std::endl;
    }
  }

  void LowStateHandler(const void *message) {
    unitree_hg::msg::dds_::LowState_ low_state =
        *(const unitree_hg::msg::dds_::LowState_ *)message;

    if (low_state.crc() !=
        Crc32Core((uint32_t *)&low_state,
                  (sizeof(unitree_hg::msg::dds_::LowState_) >> 2) - 1)) {
      std::cout << "low_state CRC Error" << std::endl;
      return;
    }
    // Emergency Stop: B
    xRockerBtnDataStruct remote_key_data;
    memcpy(&remote_key_data, &low_state.wireless_remote()[0], 40);
    if(remote_key_data.btn.components.B == 1){
      std::cout << "B button pressed: Exit" << std::endl;
      quick_exit(0);
    }

    // get motor state
    MotorState ms_tmp;
    for (int i = 0; i < G1_NUM_MOTOR; ++i) {
      ms_tmp.q.at(i) = low_state.motor_state()[i].q();
      ms_tmp.dq.at(i) = low_state.motor_state()[i].dq();
      ms_tmp.ddq.at(i) = low_state.motor_state()[i].ddq();
      ms_tmp.tau_est.at(i) = low_state.motor_state()[i].tau_est();
    }
    motor_state_buffer_.SetData(ms_tmp);

    // get imu state
    ImuState imu_tmp;
    imu_tmp.gyroscope = low_state.imu_state().gyroscope();
    imu_tmp.rpy = low_state.imu_state().rpy();
    imu_tmp.quaternion = low_state.imu_state().quaternion();
    imu_tmp.accelerometer = low_state.imu_state().accelerometer();
    imu_state_buffer_.SetData(imu_tmp);

    RCState rc_tmp;
    for (int i = 0; i < 40; ++i) {
      rc_tmp.wireless_remote.at(i) = low_state.wireless_remote().at(i);
    }
    rc_state_buffer_.SetData(rc_tmp);

    // set mode machine at first time
    if (mode_machine_ != low_state.mode_machine()) {
      if (mode_machine_ == 0)
        std::cout << "G1 type: " << unsigned(low_state.mode_machine())
                  << std::endl;
        
      mode_machine_ = low_state.mode_machine();
    }
  }
  
  void SportStateHandler(const void *message) {
    unitree_go::msg::dds_::SportModeState_ sport_state =
        *(const unitree_go::msg::dds_::SportModeState_ *)message;

    // get motor state
    SportState ss_tmp;
    ss_tmp.error_code = sport_state.error_code();
    ss_tmp.mode = sport_state.mode();
    ss_tmp.progress = sport_state.progress();
    ss_tmp.foot_raise_height = sport_state.foot_raise_height();
    ss_tmp.progress = sport_state.progress();
    
    ss_tmp.position = sport_state.position();
    
    ss_tmp.body_height = sport_state.body_height();
    ss_tmp.velocity = sport_state.velocity();
    ss_tmp.yaw_speed = sport_state.yaw_speed();

    sport_state_buffer_.SetData(ss_tmp);

  }

  void LowCommandWriter() {
    unitree_hg::msg::dds_::LowCmd_ dds_low_command;
    dds_low_command.mode_pr() = mode_;
    dds_low_command.mode_machine() = mode_machine_;

    const std::shared_ptr<const MotorCommand> motor_cmd =
        motor_command_buffer_.GetData();
    if (motor_cmd) {
      for (size_t i = 0; i < G1_NUM_MOTOR; i++) {
        dds_low_command.motor_cmd().at(i).mode() = 1;  // 1:Enable, 0:Disable
        dds_low_command.motor_cmd().at(i).tau() = motor_cmd->tau_ff.at(i);
        dds_low_command.motor_cmd().at(i).q() = motor_cmd->q_target.at(i);
        dds_low_command.motor_cmd().at(i).dq() = motor_cmd->dq_target.at(i);
        dds_low_command.motor_cmd().at(i).kp() = motor_cmd->kp.at(i);
        dds_low_command.motor_cmd().at(i).kd() = motor_cmd->kd.at(i);
      }

      dds_low_command.crc() = Crc32Core((uint32_t *)&dds_low_command,
                                        (sizeof(dds_low_command) >> 2) - 1);
      lowcmd_publisher_->Write(dds_low_command);
    }
  }

  void Control() {
    MotorCommand motor_command_tmp;
    const std::shared_ptr<const MotorState> motor_state = motor_state_buffer_.GetData();

    // init motor command by zero
    for (int i = 0; i < G1_NUM_MOTOR; ++i) {
      motor_command_tmp.tau_ff.at(i) = 0.0;
      motor_command_tmp.q_target.at(i) = 0.0;
      motor_command_tmp.dq_target.at(i) = 0.0;
      motor_command_tmp.kp.at(i) = 0.0;
      motor_command_tmp.kd.at(i) = 0.0;
    }

    if(motor_state){
      time_ += control_dt_;
      
      // set imu state
      const std::shared_ptr<const ImuState> imu_tmp = imu_state_buffer_.GetData();
      if (imu_tmp) {
        for (int i = 0; i < 3; i++){
          lcm_low_state.rpy[i] = imu_tmp->rpy.at(i);
          lcm_low_state.gyroscope[i] = imu_tmp->gyroscope.at(i);
          lcm_low_state.accelerometer[i] = imu_tmp->accelerometer.at(i);
        }
        for (int i = 0; i < 4; i++){
          lcm_low_state.quaternion[i] = imu_tmp->quaternion.at(i);
        }
      }
      // set joint state
      const std::shared_ptr<const MotorState> motor_state_tmp = motor_state_buffer_.GetData();
      if (motor_state_tmp){
        for (int i = 0; i < G1_NUM_MOTOR; ++i){
          lcm_low_state.q[i] = motor_state_tmp->q.at(i);
          lcm_low_state.dq[i] = motor_state_tmp->dq.at(i);
          lcm_low_state.ddq[i] = motor_state_tmp->ddq.at(i);
          lcm_low_state.tau_est[i] = motor_state_tmp->tau_est.at(i);
        }
      }
      // set remote controller state
      const std::shared_ptr<const RCState> rc_state_tmp = rc_state_buffer_.GetData();
      if (rc_state_tmp) {
        for (int i = 0; i < 40; ++i) {
          lcm_low_state.wireless_remote[i] = rc_state_tmp->wireless_remote.at(i);
        }
      }

      // set tick
      auto now = std::chrono::system_clock::now();
      auto timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
      lcm_low_state.tick = timestamp_ms;
      lcm_low_state.mode_machine = mode_machine_;
      lcm_low_state.mode_pr = mode_;

      _simpleLCM.publish(LCM_STATE_TOPIC, &lcm_low_state);
      
      const std::shared_ptr<const SportState> sport_tmp = sport_state_buffer_.GetData();
      
      memcpy(&lcm_sport_state.position, &sport_tmp->position, sizeof(sport_tmp->position));
      _simpleLCM.publish(LCM_SPORT_STATE_TOPIC, &lcm_sport_state);
      // lcm_sport_state = sport_tmp;
      // memcpy(&lcm_sport_state, &sport_tmp, sizeof(sport_tmp));
      // set motor command by received lcm_low_cmd
      for (int i = 0; i < G1_NUM_MOTOR; ++i) {
        motor_command_tmp.q_target.at(i) = lcm_low_cmd.q[i];    // set q_target to current state
        motor_command_tmp.dq_target.at(i) = lcm_low_cmd.dq[i];  // set dq_target to current state
        motor_command_tmp.tau_ff.at(i) = lcm_low_cmd.tau[i];    // set tau_ff to zero
        motor_command_tmp.kp.at(i) = lcm_low_cmd.kp[i];         // set kp to zero
        motor_command_tmp.kd.at(i) = lcm_low_cmd.kd[i];         // set kd to zero
      }

      motor_command_buffer_.SetData(motor_command_tmp);
    }
  }
};

int main(int argc, char const *argv[]) {
  if (argc < 2) {
    std::cout << "Usage: G1 Whole-body Control Deployment w/o hands" << std::endl;
    exit(0);
  }

  std::string networkInterface = argv[1];
  G1Control custom(networkInterface);
  while (true) usleep(20000);
  return 0;
}
