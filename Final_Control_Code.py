import math
import csv
import sys
import time
import os
from NatNetClient import NatNetClient 
import DataDescriptions
import MoCapData
import threading

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.utils import reset_estimator
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncLogger import SyncLogger
from cflib.utils import power_switch
from cflib.crazyflie.mem import MemoryElement
from cflib.crazyflie.mem import Poly4D

# ————————————————————————————————————————————————————
frame_counter = 0
cf_extpose = None  # Global variable to hold Extpos instance
print_rigid_body_data = False # Control mocap data printing for rigid body (drone)
data_lock = threading.Lock()
# ————————————————————————————————————————————————————
# Prepare CSV for to log drone mocap data and kalman states XYZ from drone
output_folder = os.path.join(os.getcwd(), "logs")
os.makedirs(output_folder, exist_ok=True)

csv_path = os.path.join(output_folder, "mocap_rigid_body_log.csv")
csv_file = open(csv_path, "w", newline="")
writer = csv.DictWriter(csv_file, fieldnames=["timestamp", "frame", "rb_id", "x", "y", "z", "qx", "qy", "qz", "qw"])
writer.writeheader()
csv_file.flush()

kalman_posvar_path = os.path.join(output_folder, "kalman_states_log.csv")
kalman_posvar_file = open(kalman_posvar_path, "w", newline="")
kalman_posvar_writer = csv.DictWriter(kalman_posvar_file, fieldnames=["timestamp", "X", "Y", "Z","VarX", "VarY", "VarZ" ])

kalman_posvar_writer.writeheader()
kalman_posvar_file.flush()

kalman_pose_path = os.path.join(output_folder, "kalman_var_log.csv")
kalman_pose_file = open(kalman_pose_path, "w", newline="")
kalman_pose_writer = csv.DictWriter(
    kalman_pose_file,
    fieldnames=["timestamp","w", "x", "y", "z" ]
)
kalman_pose_writer.writeheader(); 
kalman_pose_file.flush()


# —————————————————————————————————————————————————————

def start_posvar_logger(cf, kalman_posvar_writer, kalman_posvar_file, threshold_samples=120):

    sample_count = {"n": 0}
    ready_event = threading.Event()

    # Log config: position + attitude
    log_conf = LogConfig(name='KalmanPosVar', period_in_ms=16)
    log_conf.add_variable('kalman.stateX', 'float')
    log_conf.add_variable('kalman.stateY', 'float')
    log_conf.add_variable('kalman.stateZ', 'float')
    log_conf.add_variable('kalman.varX', 'float')
    log_conf.add_variable('kalman.varY', 'float')
    log_conf.add_variable('kalman.varZ', 'float')

    def _logger():
        try:
            with SyncLogger(cf, log_conf) as logger:
                for i, entry in enumerate(logger):
                    d = entry[1]
                    ts = time.time()
                    kalman_posvar_writer.writerow({
                        "timestamp": ts,
                        "X": d['kalman.stateX'],
                        "Y": d['kalman.stateY'],
                        "Z": d['kalman.stateZ'],
                        "VarX": d['kalman.varX'],
                        "VarY": d['kalman.varY'],
                        "VarZ": d['kalman.varZ']                                                          
                    })
                    kalman_posvar_file.flush()

                    sample_count["n"] += 1
                    if not ready_event.is_set() and sample_count["n"] >= threshold_samples:
                        ready_event.set()

        except Exception as e:
            print(f"[Logger] exception: {e}")

    print(f"Starting EKF posvar logger (waiting for {threshold_samples} samples before continuing)...")
    t = threading.Thread(target=_logger, daemon=True)
    t.start()

    ready_event.wait()  # Block until enough samples are logged
    print("Threshold reached — continuing while logging runs in background.")


def start_pose_logger(cf, kalman_pose_writer, kalman_pose_file, threshold_samples=120):
    """
    Starts EKF pose logging immediately.
    Blocks until `threshold_samples` are logged, then continues in the background
    until the program exits.
    """
    sample_count = {"n": 0}
    ready_event = threading.Event()

    # Log config: position + attitude
    log_conf = LogConfig(name='KalmanPose', period_in_ms=16)
    log_conf.add_variable('kalman.q0', 'float')
    log_conf.add_variable('kalman.q1', 'float')
    log_conf.add_variable('kalman.q2', 'float')
    log_conf.add_variable('kalman.q3', 'float')

    def _logger():
        try:
            with SyncLogger(cf, log_conf) as logger:
                for i, entry in enumerate(logger):
                    d = entry[1]
                    ts = time.time()
                    kalman_pose_writer.writerow({
                        "timestamp": ts,
                        "w": d['kalman.q0'],
                        "x": d['kalman.q1'],
                        "y": d['kalman.q2'],
                        "z": d['kalman.q3'],                                                 
                    })
                    kalman_pose_file.flush()

                    sample_count["n"] += 1
                    if not ready_event.is_set() and sample_count["n"] >= threshold_samples:
                        ready_event.set()

        except Exception as e:
            print(f"[Logger] exception: {e}")

    print(f"Starting EKF pose logger (waiting for {threshold_samples} samples before continuing)...")
    t = threading.Thread(target=_logger, daemon=True)
    t.start()

    ready_event.wait()  # Block until enough samples are logged
    print("Threshold reached — continuing while logging runs in background.")

# —————————————————————————————————————————————————————
# This is a callback function that gets connected to the NatNet client
# and called once per mocap frame.

def receive_new_frame(data_dict):
    order_list = ["frameNumber", "markerSetCount", "unlabeledMarkersCount", #type: ignore  # noqa F841
                  "rigidBodyCount", "skeletonCount", "labeledMarkerCount",
                  "timecode", "timecodeSub", "timestamp", "isRecording",
                  "trackedModelsChanged"]
    dump_args = False
    if dump_args is True:
        out_string = "    "
        for key in data_dict:
            out_string += key + "= "
            if key in data_dict:
                out_string += data_dict[key] + " "
            out_string += "/"
        print(out_string)

# —————————————————————————————————————————————————————
# This is a callback function that gets connected to the NatNet client
# and called once per mocap frame.

def receive_new_frame_with_data(data_dict):
    order_list = ["frameNumber", "markerSetCount", "unlabeledMarkersCount", #type: ignore  # noqa F841
                  "rigidBodyCount", "skeletonCount", "labeledMarkerCount",
                  "timecode", "timecodeSub", "timestamp", "isRecording",
                  "trackedModelsChanged", "offset", "mocap_data"]
    dump_args = False
    if dump_args is True:
        out_string = "    "
        for key in data_dict:
            out_string += key + "= "
            if key in data_dict:
                out_string += str(data_dict[key]) + " "
            out_string += "/"
        print(out_string)

    global frame_counter, labeled_markers, stationary_markers

    labeled_markers = {}  # Store positions of labeled markers
    stationary_markers = {} # Store stationary marker positions for waypoint selection  

    with data_lock:
        frame_counter += 1
        
        # Extract mocap data if available
        if 'mocap_data' in data_dict:
            mocap_data = data_dict['mocap_data']
            ts = time.time()
            
            # Process labeled markers (markers with names/IDs)
            if hasattr(mocap_data, 'labeled_marker_data') and mocap_data.labeled_marker_data:
                labeled_markers.clear()  # Clear previous frame data
                for marker in mocap_data.labeled_marker_data.labeled_marker_list:
                    if hasattr(marker, 'id_num') and hasattr(marker, 'pos'):
                        marker_data = {
                            "x": marker.pos[0],
                            "y": marker.pos[1],
                            "z": marker.pos[2],
                            "timestamp": ts,
                            "frame": frame_counter,
                            "size": getattr(marker, 'size', 0),
                            "residual": getattr(marker, 'residual', 0)
                        }
                        labeled_markers[marker.id_num] = marker_data
                        
                        # Store in stationary markers for waypoint selection
                        stationary_markers[marker.id_num] = {
                            "x": marker.pos[0],
                            "y": marker.pos[1],
                            "z": marker.pos[2]
                        }

# —————————————————————————————————————————————————————

# This is a callback function that gets connected to the NatNet client.
# It is called once per rigid body per frame.
# Immediately log timestamp, frame number, ID, position, and quaternion to CSV.

def receive_rigid_body_frame(new_id, position, rotation):

    global frame_counter, print_rigid_body_data
    frame_counter += 1
    x, y, z       = position
    qx, qy, qz, qw = rotation
    ts = time.time()

    row = {
        "timestamp": ts,
        "frame":     frame_counter,
        "rb_id":     new_id,
        "x":         x, "y":      y, "z":      z,
        "qx":        qx,"qy":     qy,"qz":     qz,"qw":     qw,
    }

    # write & flush csv immediately
    writer.writerow(row)
    csv_file.flush()

    # Transformation of coordinate frame to crazyflie's coordinate frame and send if cf_extpose is set
    cf_x = x
    cf_y = -z
    cf_z = y
    
    if cf_extpose:
        cf_extpose.send_extpose(cf_x,cf_y,cf_z, qx, qy, qz, qw)
        
    # Only print if explicitly enabled (for debugging)
    if print_rigid_body_data:
        print(f"[Frame {frame_counter}] RB {new_id} | Pos ({x:.2f}, {y:.2f}, {z:.2f}) | Quat ({qx:.2f}, {qy:.2f}, {qz:.2f}, {qw:.2f})")                        

# —————————————————————————————————————————————————————
# Function to list all labeled markers

def list_all_markers():
    with data_lock:
        print("\n" + "="*70)
        print("CURRENTLY TRACKED STATIONARY MARKERS:")
        print("="*70)

        if labeled_markers:
            for marker_id, pos in labeled_markers.items():
                print(f"  Marker ID {marker_id:>6}: x={pos['x']:>7.3f}, y={pos['y']:>7.3f}, z={pos['z']:>7.3f}")
        else:
            print("  No markers currently tracked")
        
        print("="*70)

# —————————————————————————————————————————————————————
# Get waypoint coordinates for labeled markers to be sent back to drone (x, y, z coordinates or none if marker not found)

def get_waypoint_from_labeled_marker(marker_id):

    if marker_id in stationary_markers:
        pos = stationary_markers[marker_id]
        return (pos["x"], pos["y"], pos["z"])
    else:
        print(f"Labeled marker {marker_id} not found or not being tracked")
        return None
    
# —————————————————————————————————————————————————————
# Function to select markers

def select_waypoint_markers():

    print("\nSelect markers for drone waypoint sequence:")
    print("Enter marker IDs in the order you want the drone to visit them.")
    print("Press Enter after each ID, type 'done' when finished, or 'cancel' to abort.")
    
    selected_markers = []
    
    while True:
        if not stationary_markers:
            print("No stationary markers available!")
            return []
            
        # Show available markers
        print(f"\nAvailable markers: {list(stationary_markers.keys())}")
        print(f"Currently selected: {selected_markers}")
        
        user_input = input("Enter marker ID (or 'done'/'cancel'): ").strip()
        
        if user_input.lower() == 'done':
            break
        elif user_input.lower() == 'cancel':
            return []
        
        try:                                                     #TEST: DISABLE TO SEE IF CAN ADD MARKERS OVER AND OVER
            marker_id = int(user_input)
            if marker_id in stationary_markers:
                selected_markers.append(marker_id)
                pos = stationary_markers[marker_id]
                print(f"Added marker {marker_id} at position ({pos['x']:.3f}, {pos['y']:.3f}, {pos['z']:.3f})")
            else:
                print(f"Marker {marker_id} not found in available markers!")
        except ValueError:
            print("Please enter a valid marker ID number!")
    
    return selected_markers

# —————————————————————————————————————————————————————
# Create a simple polynomial trajectory between two points

def generate_simple_polynomial(start_pos, end_pos, duration=2.0):

    # Simple 5th order polynomial for smooth trajectory
    # Position: s(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
    # Velocity: s'(t) = a1 + 2*a2*t + 3*a3*t^2 + 4*a4*t^3 + 5*a5*t^4
    # Acceleration: s''(t) = 2*a2 + 6*a3*t + 12*a4*t^2 + 20*a5*t^3
    
    # Boundary conditions: start at rest, end at rest
    # s(0) = start_pos, s'(0) = 0, s''(0) = 0
    # s(T) = end_pos, s'(T) = 0, s''(T) = 0
    
    T = duration
    dx = end_pos[0] - start_pos[0]
    dy = end_pos[1] - start_pos[1] 
    dz = end_pos[2] - start_pos[2]
    
    # Coefficients for 5th order polynomial with zero velocity/acceleration at endpoints
    x_coeffs = [start_pos[0], 0, 0, 10*dx/(T**3), -15*dx/(T**4), 6*dx/(T**5), 0, 0]
    y_coeffs = [start_pos[1], 0, 0, 10*dy/(T**3), -15*dy/(T**4), 6*dy/(T**5), 0, 0]
    z_coeffs = [start_pos[2], 0, 0, 10*dz/(T**3), -15*dz/(T**4), 6*dz/(T**5), 0, 0]
    yaw_coeffs = [0, 0, 0, 0, 0, 0, 0, 0]  # Keep yaw constant
    
    return [duration] + x_coeffs + y_coeffs + z_coeffs + yaw_coeffs

# —————————————————————————————————————————————————————
# Create trajectory using selected marker position

def create_trajectory_from_markers(marker_ids, segment_duration=3.0):

    if len(marker_ids) < 2:
        print("Need at least 2 markers to create a trajectory!")
        return None
        
    trajectory = []
    
    print(f"\nGenerating trajectory through {len(marker_ids)} markers:")
    
    for i in range(len(marker_ids) - 1):
        start_marker = marker_ids[i]
        end_marker = marker_ids[i + 1]
        
        # Get start and end marker position coordinates
        start_pos = get_waypoint_from_labeled_marker(start_marker)
        end_pos = get_waypoint_from_labeled_marker(end_marker)
        
        if start_pos is None or end_pos is None:
            print(f"Error: Could not get position for markers {start_marker} or {end_marker}")
            return None
            
        # Transform coordinates for drone 
        start_drone = (start_pos[0], -start_pos[2], start_pos[1]+1.0)
        end_drone = (end_pos[0], -end_pos[2], end_pos[1]+1.0)
        
        # Generate polynomial segment using function
        segment = generate_simple_polynomial(start_drone, end_drone, segment_duration)
        trajectory.append(segment)
        
        print(f"  Segment {i+1}: Marker {start_marker} → Marker {end_marker}")
        print(f"    Start: ({start_drone[0]:.2f}, {start_drone[1]:.2f}, {start_drone[2]:.2f})")
        print(f"    End:   ({end_drone[0]:.2f}, {end_drone[1]:.2f}, {end_drone[2]:.2f})")
        print(f"    Duration: {segment_duration:.1f}s")
    
    # Calculate total duration for full trajectory
    total_duration = len(trajectory) * segment_duration
    print(f"\nTotal trajectory duration: {total_duration:.1f} seconds")
    
    return trajectory

# —————————————————————————————————————————————————————
# Upload trajectory to crazyflie memory

def upload_trajectory(cf, trajectory_id, trajectory):
    trajectory_mem = cf.mem.get_mems(MemoryElement.TYPE_TRAJ)[0]
    trajectory_mem.trajectory = []

    total_duration = 0
    for row in trajectory:
        duration = row[0]
        x = Poly4D.Poly(row[1:9])
        y = Poly4D.Poly(row[9:17])
        z = Poly4D.Poly(row[17:25])
        yaw = Poly4D.Poly(row[25:33])
        trajectory_mem.trajectory.append(Poly4D(duration, x, y, z, yaw))
        total_duration += duration

    print("Uploading trajectory to Crazyflie")
    trajectory_mem.write_data_sync()
    cf.high_level_commander.define_trajectory(trajectory_id, 0, len(trajectory_mem.trajectory))
    print("Trajectory uploaded successfully")
    return total_duration

# —————————————————————————————————————————————————————
# Go to first marker in selected markers

def goto_start_trajectory(cf, selected_markers):
    if not selected_markers:
        print("No markers selected!")
        return
    
    first_id = selected_markers[0]
    wp = get_waypoint_from_labeled_marker(first_id)
    if wp is None:
        print(f"Marker {first_id} not found in stationary_markers")
        return
    

    x_opt, y_opt, z_opt = wp
    cf_startx = x_opt
    cf_starty = -z_opt
    cf_startz = y_opt + 1.0

    cf.high_level_commander.go_to(cf_startx, cf_starty, cf_startz, yaw=0.0, duration_s = 2.0)
                          

# —————————————————————————————————————————————————————
# Class to setup sending of external position back to crazyflie drone

class Extpos():
    def __init__(self, crazyflie=None):
        """
        Initialize the Extpos object.
        """
        self._cf = crazyflie

    def send_extpose(self, x, y, z, qx, qy, qz, qw):
        """
        Send the current Crazyflie X, Y, Z position and attitude as a
        normalized quaternion. This is going to be forwarded to the
        Crazyflie's position estimator.
        """
        self._cf.loc.send_extpose([x, y, z], [qx, qy, qz, qw])

# —————————————————————————————————————————————————————
# Function to define run sequence for flying to markers

def run_sequence_to_markers(cf, marker_ids):
    commander = cf.high_level_commander
   
    print("\n" + "="*50)
    print("STARTING DRONE SEQUENCE")
    print("="*50)
    
    # Arm and takeoff
    print("Arming drone")
    cf.platform.send_arming_request(True)
    time.sleep(2.0)

    print("Taking off")
    commander.takeoff(1.0, 5.0)
    time.sleep(5.0)

    # Visit each marker in order
    for i, marker_id in enumerate(marker_ids, 1):
        print(f"\n Waypoint {i}/{len(marker_ids)}: Going to Marker {marker_id}")
        
        waypoint = get_waypoint_from_labeled_marker(marker_id)
        if waypoint is None:
            print(f"   ERROR: Marker {marker_id} not found, skipping...")
            continue
            
        a, b, c = waypoint
        
        # Transform coordinates for drone
        drone_x = a
        drone_y = -c  
        drone_z = b+1.0

        print(f"   Target position: ({drone_x:.2f}, {drone_y:.2f}, {drone_z:.2f})")
        
        commander.go_to(drone_x, drone_y, drone_z, yaw=0.0, duration_s=5.0)
        time.sleep(4.0)  # Wait for motion to complete
        
        print(f" Reached waypoint {i}")

    # Land
    print("\n Landing...")
    commander.land(-1.0, 5.0)
    time.sleep(3)
    commander.stop()
    
    print("Sequence completed!")
    print("="*50)

# —————————————————————————————————————————————————————
# Function to define sequence to fly trajectory
def run_trajectory_sequence(cf, trajectory_id, duration):
    commander = cf.high_level_commander
    
    print("\n" + "="*50)
    print("STARTING TRAJECTORY EXECUTION")
    print("="*50)

    print("Arming drone")
    cf.platform.send_arming_request(True)
    time.sleep(2.0)

    print("Taking off")
    commander.takeoff(1.0, 3.0)
    time.sleep(4.0)

    print("Going to starting point of trajectory")
    goto_start_trajectory(cf, selected_markers)
    time.sleep(4.0)
    
    print("Starting trajectory")
    relative = False
    commander.start_trajectory(trajectory_id, 1.0, relative)

    # Show progress during trajectory
    start_time = time.time()
    while time.time() - start_time < duration:
        elapsed = time.time() - start_time
        progress = (elapsed / duration) * 100
        print(f"Trajectory progress: {progress:.1f}% ({elapsed:.1f}s / {duration:.1f}s)")
        time.sleep(2.0)
    
    print("Trajectory completed")
    time.sleep(1.0)

    print("Landing")
    commander.land(0.0, 3.0)
    time.sleep(3)
    commander.stop()
    
    print("Sequence completed!")
    print("="*50)


#-------------

def hover(cf):
    commander = cf.high_level_commander
   
    cf.platform.send_arming_request(True)
    time.sleep(2.0)

    #takeoff_yaw = 3.14 / 2 if relative_yaw else 0.0
    commander.takeoff(0.5, 5.0)
    time.sleep(30.0)
    commander.land(-0.5, 5.0)
    time.sleep(3)
    commander.stop()

# —————————————————————————————————————————————————————
# Function to restart STM32 of crazyflie

def try_stm_power_cycle(uri):
    try:
        print("Attempting STM32 power cycle")
        ps = power_switch.PowerSwitch(uri)
        ps.stm_power_cycle()
        ps.close()
        print("STM32 power cycle completed")
        return True
    except Exception as e:
        print(f"stm_power_cycle() failed: {e}")
        sys.exit(1)
        return False
    
# —————————————————————————————————————————————————————
# Function to power down STM32 of crazyflie

def power_down(uri):
    try:
        print("Attempting STM32 power down")
        ps = power_switch.PowerSwitch(uri)
        ps.stm_power_down()
        ps.close()
        print("STM32 power down completed")
        return True
    except Exception as e:
        print(f"stm_power_down() failed: {e}")
        sys.exit(1)
        return False   
    
    
def add_lists(totals, totals_tmp):
    totals[0] += totals_tmp[0]
    totals[1] += totals_tmp[1]
    totals[2] += totals_tmp[2]
    return totals

def print_configuration(natnet_client):
    natnet_client.refresh_configuration()
    print("Connection Configuration:")
    print("  Client:       %s" % natnet_client.local_ip_address)
    print("  Server:       %s" % natnet_client.server_ip_address)
    print("  Command Port: %d" % natnet_client.command_port)
    print("  Data Port:    %d" % natnet_client.data_port)
    if natnet_client.use_multicast:
        print("  Using Multicast")
        print("  Multicast Group: %s" % natnet_client.multicast_address)
    else:
        print("  Using Unicast")

    #NatNet Server Info
    app_name = natnet_client.get_application_name()
    srv_ver  = natnet_client.get_server_version()
    nn_ver_r = natnet_client.get_nat_net_requested_version()
    nn_ver_s = natnet_client.get_nat_net_version_server()
    
    print(f"  Application: {app_name}")
    print(f"  Server Version: {srv_ver}")
    print(f"  Requested NatNet Version: {nn_ver_r}")
    print(f"  Server NatNet Version:    {nn_ver_s}")
    print("  Python Version: %s" % sys.version)

def print_commands(can_change_bitstream):
    s = (
        "Commands:\n"
        "  s: send data descriptions\n"
        "  r: resume/start frame playback\n"
        "  p: pause frame playback\n"
        "  o: reset working range\n"
        "  w: set working range\n"
        "  m: show current markers\n"
        "  t: create and execute trajectory (select waypoints)\n"
        "  f: start flight sequence (select waypoints)\n"
        "  d: toggle rigid body debug printing\n"
        "  c: print configuration\n"
        "  h: print commands\n"
        "  q: quit\n"
    )
    print(s)

def request_data_descriptions(s_client):
    s_client.send_request(
        s_client.command_socket,
        s_client.NAT_REQUEST_MODELDEF,
        "",
        (s_client.server_ip_address, s_client.command_port)
    )

def test_classes():
    totals = [0, 0, 0]
    print("Test Data Description Classes")
    totals = add_lists(totals, DataDescriptions.test_all())
    print("\nTest MoCap Frame Classes")
    totals = add_lists(totals, MoCapData.test_all())
    print("\nAll Tests totals")
    print("--------------------")
    print(f"[PASS] {totals[0]}\n[FAIL] {totals[1]}\n[SKIP] {totals[2]}")

def my_parse_args(arg_list, args_dict):
    if len(arg_list) > 1: args_dict["serverAddress"] = arg_list[1]
    if len(arg_list) > 2: args_dict["clientAddress"] = arg_list[2]
    if len(arg_list) > 3: args_dict["use_multicast"] = not arg_list[3].upper().startswith("U")
    if len(arg_list) > 4: args_dict["stream_type"] = arg_list[4]
    return args_dict

if __name__ == "__main__":
    try:
        # Initialize the Crazyflie connection
        cflib.crtp.init_drivers()
        uri = 'radio://0/80/2M/E7E7E7E7E1'  # Or your custom URI
        print("Connecting to Crazyflie...")
      
        try_stm_power_cycle(uri)

        print ('Waiting for STM32 to boot')
        time.sleep(5)

        print ('Connecting via SyncCrazyflie')
        scf = SyncCrazyflie(uri, cf=Crazyflie(rw_cache='./cache'))
        try:
            scf.__enter__()
            print('Connected to Crazyflie')
        except Exception as e:
            print(f"Failed to connect to Crazyflie: {e}")
            sys.exit(1)
        cf = scf.cf

        print('NatNet Client Setup)')
        optionsDict = {
            "clientAddress": "127.0.0.1",
            "serverAddress": "127.0.0.1",
            "use_multicast": None,
            "stream_type":   None
        }
        optionsDict = my_parse_args(sys.argv, optionsDict)
        streaming_client = NatNetClient()
        streaming_client.set_client_address(optionsDict["clientAddress"])
        streaming_client.set_server_address(optionsDict["serverAddress"])
        #streaming_client.new_frame_listener      = receive_new_frame
        streaming_client.rigid_body_listener    = receive_rigid_body_frame
        streaming_client.new_frame_with_data_listener = receive_new_frame_with_data

        print("NatNet Python Client 4.3\n")

        # Choose Multicast or Unicast
        cast_choice = int(input("Select 0 for multicast and 1 for unicast: "))
        optionsDict["use_multicast"] = (cast_choice == 0)
        streaming_client.set_use_multicast(optionsDict["use_multicast"])

        client_addr_choice = input("Client Address (127.0.0.1): ")
        if client_addr_choice:
            streaming_client.set_client_address(client_addr_choice)
        server_addr_choice = input("Server Address (127.0.0.1): ")
        if server_addr_choice:
            streaming_client.set_server_address(server_addr_choice)

        # Choose data vs command stream
        stream_choice = None
        while stream_choice not in ('d','c'):
            stream_choice = input("Select d for datastream and c for command stream: ").lower()
        optionsDict["stream_type"] = stream_choice

        # Start streaming
        is_running = streaming_client.run(optionsDict["stream_type"])
        if not is_running:
            print("ERROR: Could not start streaming client."); sys.exit(1)

        # Suppress Motive's default prints
        streaming_client.set_print_level(0)

        cf_extpose = Extpos(cf)  # Assign global instance
        print(f"cf_extpose is {'set' if cf_extpose else 'NOT set'}")

        # Activate Kalman estimator and reset
        print("\nConfiguring Kalman estimator...")
        cf.param.set_value('stabilizer.estimator', '2') #
        cf.param.set_value('locSrv.extPosStdDev', '0.003') #0.001-0.1 
        cf.param.set_value('locSrv.extQuatStdDev', '0.14') #0.03-0.1 
        cf.param.set_value('kalman.mNGyro_yaw', '0.02') #default 0.1

        print("Sending pose data to drone for Kalman estimator...")
        time.sleep(2.0)  # Let a few pose packets stream in
        print("Resetting Kalman estimator...")
        reset_estimator.reset_estimator(cf)
        print("Estimator reset complete.\n")
        time.sleep(5.0)

        start_posvar_logger(cf, kalman_posvar_writer, kalman_posvar_file, threshold_samples=120)
        start_pose_logger(cf, kalman_pose_writer, kalman_pose_file, threshold_samples=120)

        # Interactive command loop
        print("System ready! Use 'f' to start flight sequence, 'm' to show markers, 't' to start trajectory, 'h' for help, 'q' for hover")
        is_looping = True
        while is_looping:
            cmd = input("\n(h for help)> ").strip().lower()
            if not cmd:
                continue
            if cmd == 'h':
                print_commands(streaming_client.can_change_bitstream_version())
            elif cmd == 'c':
                print_configuration(streaming_client)
            elif cmd == 'm':
                list_all_markers()
                
            elif cmd == 'f':
                # Flight sequence
                list_all_markers()
                selected_markers = select_waypoint_markers()
                if selected_markers:
                    print(f"\nExecuting flight sequence to markers: {selected_markers}")
                    run_sequence_to_markers(cf, selected_markers)
                else:
                    print("Flight sequence cancelled.")

            elif cmd == 't':
                # Trajectory creation and execution
                list_all_markers()
                selected_markers = select_waypoint_markers()
                if selected_markers:
                    print(f"\nSelected markers: {selected_markers}")
                    
                    # Set segment duration
                    try:
                        duration_input = input("Segment duration (seconds, default 3.0): ").strip()
                        segment_duration = float(duration_input) if duration_input else 3.0
                    except ValueError:
                        segment_duration = 3.0
                        print("Invalid input, using default duration of 3.0 seconds")
                    
                    # Generate trajectory
                    trajectory = create_trajectory_from_markers(selected_markers, segment_duration)
                    if trajectory:
                        try:
                            trajectory_id = 1
                            total_duration = upload_trajectory(cf, trajectory_id, trajectory)
                            
                            confirm = input(f"\nExecute trajectory? ({total_duration:.1f}s total) (y/n): ").strip().lower()
                            if confirm == 'y':
                                run_trajectory_sequence(cf, trajectory_id, total_duration)
                            else:
                                print("Trajectory cancelled.")
                        except Exception as e:
                            print(f"Error during trajectory execution: {e}")
                    else:
                        print("Failed to generate trajectory." \
                        "")
                else:
                    print("Trajectory creation cancelled.")

            elif cmd == 'q':
                hover(cf)
            elif cmd == 'd':
                print_rigid_body_data = not print_rigid_body_data
                print(f"Rigid body debug printing: {'ON' if print_rigid_body_data else 'OFF'}")
            elif cmd == 's':
                request_data_descriptions(streaming_client)
                time.sleep(1)
            elif cmd in ('3','4'):
                if streaming_client.can_change_bitstream_version():
                    major, minor = (3,1) if cmd=='3' else (4,1)
                    rc = streaming_client.set_nat_net_version(major, minor)
                    time.sleep(1)
                    print(f"Set NatNet version to {major}.{minor}, return code {rc}")
                else:
                    print("Can only change bitstream in Unicast mode.")
            elif cmd == 'p':
                streaming_client.send_command("TimelineStop"); time.sleep(1)
            elif cmd == 'r':
                streaming_client.send_command("TimelinePlay")
            elif cmd == 'q':
                power_down(uri)
                is_looping = False
            elif cmd == 't':
                test_classes()
            else:
                print(f"Unknown command '{cmd}'. Type 'h' for help.")

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        # Clean up
        csv_file.close()
        kalman_posvar_file.close()
        kalman_pose_file.close()
        streaming_client.shutdown()
        scf.__exit__(None, None, None)  # Close Crazyflie connection
        print("Exiting.")