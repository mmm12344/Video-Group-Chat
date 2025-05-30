import tkinter as tk
from tkinter import messagebox, simpledialog, Label, Text, Entry, Button, Scrollbar, END
import socket
import threading
import pickle
import time
import cv2
import pyaudio
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont
import struct
import queue
import os

# Constants
TCP_PORT = 65000
VIDEO_UDP_PORT = 65001
AUDIO_UDP_PORT = 65002
BUFFER_SIZE = 65536  # Max UDP packet size

# Audio Constants
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 22050

# Video Constants
VIDEO_WIDTH = 320
VIDEO_HEIGHT = 240
VIDEO_QUALITY = 80

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = "127.0.0.1"
    finally:
        s.close()
    return IP

class VideoChatApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Local Video Chat")
        self.root.geometry("1000x700")

        self.current_frame = None
        self.username = None
        self.is_host = False
        self.host_ip = None
        self.my_ip = get_local_ip()

        self.tcp_socket = None
        self.udp_video_socket = None
        self.udp_audio_socket = None
        # Store client info: {ip: {"username": name, "conn": conn_obj, "mic_on": True, "cam_on": True, "video_label": label, "name_label": label}}
        self.clients = {}
        self.client_connections = {}  # Host only: {ip: conn}
        self.running = False
        self.threads = []

        self.mic_enabled = True
        self.cam_enabled = True

        self.p_audio = None
        self.audio_stream_in = None
        self.audio_stream_out = None
        self.video_capture = None
        self.video_labels = {}  # {ip: Label widget for video}
        self.name_labels = {}  # {ip: Label widget for name/status}
        self.self_video_label = None
        self.self_name_label = None
        self.chat_display = None
        self.chat_input = None
        self.audio_queue = queue.Queue()

        # Placeholder image for when camera is off
        self.cam_off_image = self.create_placeholder_image("Camera Off")

        self.show_initial_screen()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_placeholder_image(self, text):
        img = Image.new("RGB", (VIDEO_WIDTH, VIDEO_HEIGHT), color="black")
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.load_default()
            text_width = draw.textlength(text, font=font)
            text_height = font.size
            position = ((VIDEO_WIDTH - text_width) // 2, (VIDEO_HEIGHT - text_height) // 2)
            draw.text(position, text, fill="white", font=font)
        except Exception:
            draw.text((10, 10), text, fill="white")
        return ImageTk.PhotoImage(image=img)

    def clear_frame(self):
        if self.current_frame:
            # Only destroy the frame, don't stop media streams
            self.current_frame.destroy()
            self.video_labels = {}
            self.name_labels = {}
            self.self_video_label = None
            self.self_name_label = None
            self.chat_display = None
            self.chat_input = None

    def show_initial_screen(self):
        self.clear_frame()
        self.running = False
        self.close_sockets()
        self.stop_media_streams()

        current_threads = self.threads[:]
        self.threads = []
        for t in current_threads:
            if t.is_alive():
                try:
                    t.join(timeout=0.2)
                except RuntimeError:
                    pass

        self.clients = {}
        self.client_connections = {}
        self.is_host = False
        self.host_ip = None
        self.username = None

        self.current_frame = tk.Frame(self.root)
        self.current_frame.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)

        tk.Label(self.current_frame, text="Welcome to Local Video Chat!", font=("Arial", 16)).pack(pady=10)
        tk.Label(self.current_frame, text=f"Your IP: {self.my_ip}", font=("Arial", 10)).pack(pady=5)

        tk.Button(self.current_frame, text="Start a New Group Chat", command=self.start_chat_prompt).pack(pady=5)
        tk.Button(self.current_frame, text="Join an Existing Chat", command=self.join_chat_prompt).pack(pady=5)

    def start_chat_prompt(self):
        username = simpledialog.askstring("Username", "Enter your username:", parent=self.root)
        if username:
            self.username = username
            self.is_host = True
            self.host_ip = self.my_ip
            print(f"Starting chat as host {self.username} ({self.my_ip})...")
            if self.setup_networking() and self.setup_media():
                self.show_chat_screen()
                self.start_media_streams()  # Start media AFTER setting up GUI
            else:
                messagebox.showerror("Setup Error", "Failed to set up server sockets or media devices. Check ports and permissions.")
                self.show_initial_screen()
        else:
            messagebox.showwarning("Input Error", "Username cannot be empty.")

    def join_chat_prompt(self):
        username = simpledialog.askstring("Username", "Enter your username:", parent=self.root)
        if not username:
            messagebox.showwarning("Input Error", "Username cannot be empty.")
            return

        host_ip_to_join = simpledialog.askstring("Join Chat", "Enter the Host IP address:", parent=self.root)
        if host_ip_to_join:
            self.username = username
            self.is_host = False
            self.host_ip = host_ip_to_join
            print(f"Attempting to join chat at {self.host_ip} as {self.username}...")
            if self.setup_networking() and self.setup_media():
                self.show_chat_screen()
                self.start_media_streams()  # Start media AFTER setting up GUI
            else:
                messagebox.showerror("Setup Error", f"Failed to connect to host {self.host_ip} or setup media. Check IP and host status.")
                self.show_initial_screen()
        else:
            messagebox.showwarning("Input Error", "Host IP cannot be empty.")

    def setup_networking(self):
        self.running = True
        try:
            # UDP sockets
            self.udp_video_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.udp_video_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.udp_video_socket.bind(("", VIDEO_UDP_PORT))
            self.udp_video_socket.settimeout(1.0)

            self.udp_audio_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.udp_audio_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.udp_audio_socket.bind(("", AUDIO_UDP_PORT))
            self.udp_audio_socket.settimeout(1.0)

            # TCP socket
            self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.tcp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.tcp_socket.settimeout(5.0)

            if self.is_host:
                self.tcp_socket.bind((self.my_ip, TCP_PORT))
                self.tcp_socket.listen(5)
                print(f"Host listening on {self.my_ip}:{TCP_PORT}")
                tcp_thread = threading.Thread(target=self.host_listen_tcp, daemon=True)
                self.threads.append(tcp_thread)
                tcp_thread.start()
                # Add host itself to clients list
                self.clients[self.my_ip] = {
                    "username": self.username,
                    "conn": None,
                    "mic_on": self.mic_enabled,
                    "cam_on": self.cam_enabled
                }
            else:
                self.tcp_socket.connect((self.host_ip, TCP_PORT))
                print(f"Connected to host {self.host_ip}:{TCP_PORT}")
                join_msg = {
                    "type": "join",
                    "username": self.username,
                    "mic_on": self.mic_enabled,
                    "cam_on": self.cam_enabled
                }
                self.send_tcp_message(pickle.dumps(join_msg))
                tcp_thread = threading.Thread(target=self.client_listen_tcp, daemon=True)
                self.threads.append(tcp_thread)
                tcp_thread.start()

            # Start UDP listeners
            udp_video_thread = threading.Thread(target=self.listen_udp_video, daemon=True)
            self.threads.append(udp_video_thread)
            udp_video_thread.start()

            udp_audio_thread = threading.Thread(target=self.listen_udp_audio, daemon=True)
            self.threads.append(udp_audio_thread)
            udp_audio_thread.start()

            return True
        except socket.error as e:
            print(f"Socket error during setup: {e}")
            self.close_sockets()
            self.running = False
            return False
        except Exception as e:
            print(f"Unexpected error during setup: {e}")
            self.close_sockets()
            self.running = False
            return False

    def setup_media(self):
        try:
            # Audio setup
            self.p_audio = pyaudio.PyAudio()
            self.audio_stream_in = self.p_audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK
            )
            self.audio_stream_out = self.p_audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                output=True,
                frames_per_buffer=CHUNK
            )
            print("Audio devices opened successfully.")

            # Video setup
            # Try to open camera with default backend first
            self.video_capture = cv2.VideoCapture(0)
            if not self.video_capture.isOpened():
                # Try DirectShow backend for Windows if default fails
                if os.name == 'nt':
                    self.video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                if not self.video_capture.isOpened():
                    print("Error: Cannot open camera")
                    messagebox.showerror("Camera Error", "Could not open webcam. Check connection and permissions.")
                    return False
                    
            self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_WIDTH)
            self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_HEIGHT)
            print("Video capture opened successfully.")
            return True
        except Exception as e:
            print(f"Error setting up media devices: {e}")
            messagebox.showerror("Media Error", f"Could not initialize audio/video devices: {e}")
            self.stop_media_streams()
            return False

    def start_media_streams(self):
        if not self.running:
            return

        if self.cam_enabled:
            video_send_thread = threading.Thread(target=self.send_video_frames, daemon=True)
            self.threads.append(video_send_thread)
            video_send_thread.start()

        if self.mic_enabled:
            audio_send_thread = threading.Thread(target=self.send_audio_chunks, daemon=True)
            self.threads.append(audio_send_thread)
            audio_send_thread.start()

        # Start audio playback thread
        audio_play_thread = threading.Thread(target=self.play_audio, daemon=True)
        self.threads.append(audio_play_thread)
        audio_play_thread.start()

    def stop_media_streams(self):
        print("Stopping media streams...")
        # Release video capture
        if self.video_capture and self.video_capture.isOpened():
            self.video_capture.release()
            print("Video capture released.")
        self.video_capture = None

        # Close audio streams
        if self.audio_stream_in and self.audio_stream_in.is_active():
            self.audio_stream_in.stop_stream()
            self.audio_stream_in.close()
            print("Audio input stream closed.")
        self.audio_stream_in = None

        if self.audio_stream_out and self.audio_stream_out.is_active():
            self.audio_stream_out.stop_stream()
            self.audio_stream_out.close()
            print("Audio output stream closed.")
        self.audio_stream_out = None

        # Terminate PyAudio
        if self.p_audio:
            self.p_audio.terminate()
            print("PyAudio terminated.")
        self.p_audio = None

        # Clear audio queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

    # --- TCP Handling ---
    def host_listen_tcp(self):
        while self.running:
            try:
                conn, addr = self.tcp_socket.accept()
                ip = addr[0]  # Use IP as identifier
                print(f"Accepted connection from {ip}")
                self.client_connections[ip] = conn
                client_handler_thread = threading.Thread(
                    target=self.handle_client_tcp,
                    args=(conn, ip),
                    daemon=True
                )
                client_handler_thread.start()
                self.threads.append(client_handler_thread)
            except socket.timeout:
                continue
            except socket.error as e:
                if self.running:
                    print(f"Host TCP accept error: {e}")
                break
            except Exception as e:
                if self.running:
                    print(f"Unexpected error in host_listen_tcp: {e}")
                break
        print("Host TCP listening thread stopped.")

    def handle_client_tcp(self, conn, ip):
        client_username = None
        client_initial_mic_on = True
        client_initial_cam_on = True
        try:
            while self.running:
                data = conn.recv(4096)
                if not data:
                    break

                message = pickle.loads(data)

                if message["type"] == "join":
                    client_username = message["username"]
                    client_initial_mic_on = message.get("mic_on", True)
                    client_initial_cam_on = message.get("cam_on", True)
                    
                    self.clients[ip] = {
                        "username": client_username,
                        "conn": conn,
                        "mic_on": client_initial_mic_on,
                        "cam_on": client_initial_cam_on
                    }
                    
                    print(f"User {client_username} joined from {ip} (Mic: {client_initial_mic_on}, Cam: {client_initial_cam_on})")
                    
                    # Send current client list to new client
                    current_clients_info = {
                        ip: {"username": d["username"], "mic_on": d["mic_on"], "cam_on": d["cam_on"]}
                        for ip, d in self.clients.items()
                    }
                    self.send_tcp_message(
                        pickle.dumps({
                            "type": "client_list",
                            "clients": current_clients_info
                        }),
                        specific_conn=conn
                    )
                    
                    # Notify other clients
                    self.broadcast_tcp({
                        "type": "user_join",
                        "username": client_username,
                        "ip": ip,
                        "mic_on": client_initial_mic_on,
                        "cam_on": client_initial_cam_on
                    }, exclude_ip=ip)
                    
                    # Update GUI
                    self.root.after(0, self.add_video_label, ip, client_username, client_initial_mic_on, client_initial_cam_on)
                    self.root.after(0, self.display_chat_message, f"--- {client_username} joined the chat ---")

                elif message["type"] == "leave":
                    break
                    
                elif message["type"] == "mic_status":
                    enabled = message["enabled"]
                    if ip in self.clients:
                        self.clients[ip]["mic_on"] = enabled
                        print(f"Mic status update from {client_username}: {enabled}")
                        self.root.after(0, self.update_status_indicator, ip)
                        # Relay status to other clients
                        self.broadcast_tcp({
                            "type": "mic_status",
                            "ip": ip,
                            "enabled": enabled
                        }, exclude_ip=ip)
                        
                elif message["type"] == "cam_status":
                    enabled = message["enabled"]
                    if ip in self.clients:
                        self.clients[ip]["cam_on"] = enabled
                        print(f"Cam status update from {client_username}: {enabled}")
                        self.root.after(0, self.update_status_indicator, ip)
                        # Relay status to other clients
                        self.broadcast_tcp({
                            "type": "cam_status",
                            "ip": ip,
                            "enabled": enabled
                        }, exclude_ip=ip)
                        # Update video display
                        if not enabled:
                            self.root.after(0, self.show_cam_off_placeholder, ip)
                            
                elif message["type"] == "chat":
                    chat_text = message["text"]
                    sender_name = message["username"]
                    print(f"Chat from {sender_name}: {chat_text}")
                    self.root.after(0, self.display_chat_message, f"{sender_name}: {chat_text}")
                    self.broadcast_tcp({
                        "type": "chat",
                        "username": sender_name,
                        "text": chat_text
                    }, exclude_ip=ip)
                    
        except (pickle.UnpicklingError, KeyError, TypeError, EOFError) as e:
            print(f"Error decoding/processing message from {ip}: {e}")
        except socket.error as e:
            print(f"Socket error with client {ip}: {e}")
        except Exception as e:
            print(f"Unexpected error handling client {ip}: {e}")
        finally:
            if ip in self.clients:
                client_username = self.clients[ip]["username"]
                del self.clients[ip]
                print(f"User {client_username} ({ip}) disconnected.")
                self.broadcast_tcp({
                    "type": "user_leave",
                    "username": client_username,
                    "ip": ip
                })
                self.root.after(0, self.remove_video_label, ip)
                self.root.after(0, self.display_chat_message, f"--- {client_username} left the chat ---")
                
            if ip in self.client_connections:
                del self.client_connections[ip]
                
            try:
                conn.close()
            except:
                pass
            print(f"Closed connection with {ip}")

    def client_listen_tcp(self):
        while self.running:
            try:
                data = self.tcp_socket.recv(4096)
                if not data:
                    print("Server disconnected.")
                    self.handle_disconnection("Lost connection to the host.")
                    break

                message = pickle.loads(data)

                if message["type"] == "client_list":
                    self.clients = message["clients"]
                    print(f"Received initial client list: {len(self.clients)} users")
                    self.root.after(0, self.update_client_video_labels)
                    
                elif message["type"] == "user_join":
                    ip = message["ip"]
                    username = message["username"]
                    mic_on = message.get("mic_on", True)
                    cam_on = message.get("cam_on", True)
                    self.clients[ip] = {
                        "username": username,
                        "mic_on": mic_on,
                        "cam_on": cam_on
                    }
                    print(f"User {username} joined.")
                    self.root.after(0, self.add_video_label, ip, username, mic_on, cam_on)
                    self.root.after(0, self.display_chat_message, f"--- {username} joined the chat ---")
                    
                elif message["type"] == "user_leave":
                    ip = message["ip"]
                    if ip in self.clients:
                        left_username = self.clients.pop(ip)["username"]
                        print(f"User {left_username} left.")
                        self.root.after(0, self.remove_video_label, ip)
                        self.root.after(0, self.display_chat_message, f"--- {left_username} left the chat ---")
                        
                elif message["type"] == "mic_status":
                    ip = message["ip"]
                    enabled = message["enabled"]
                    if ip in self.clients:
                        self.clients[ip]["mic_on"] = enabled
                        uname = self.clients[ip]["username"]
                        print(f"Mic status update from {uname}: {enabled}")
                        self.root.after(0, self.update_status_indicator, ip)
                        
                elif message["type"] == "cam_status":
                    ip = message["ip"]
                    enabled = message["enabled"]
                    if ip in self.clients:
                        self.clients[ip]["cam_on"] = enabled
                        uname = self.clients[ip]["username"]
                        print(f"Cam status update from {uname}: {enabled}")
                        self.root.after(0, self.update_status_indicator, ip)
                        if not enabled:
                            self.root.after(0, self.show_cam_off_placeholder, ip)
                            
                elif message["type"] == "chat":
                    chat_text = message["text"]
                    sender_name = message["username"]
                    self.root.after(0, self.display_chat_message, f"{sender_name}: {chat_text}")
                    
                elif message["type"] == "host_shutdown":
                    print("Host is shutting down.")
                    self.handle_disconnection("Host has ended the chat.")
                    break
                    
            except socket.timeout:
                continue
            except (pickle.UnpicklingError, KeyError, TypeError, EOFError) as e:
                print(f"Error decoding/processing message from server: {e}")
            except socket.error as e:
                if self.running:
                    print(f"Client TCP receive error: {e}")
                    self.handle_disconnection("Lost connection to the host.")
                break
            except Exception as e:
                if self.running:
                    print(f"Unexpected error in client_listen_tcp: {e}")
                break
        print("Client TCP listening thread stopped.")

    def send_tcp_message(self, message_bytes, specific_conn=None):
        if not self.running:
            return
            
        try:
            if self.is_host and specific_conn:
                specific_conn.sendall(message_bytes)
            else:
                self.tcp_socket.sendall(message_bytes)
        except socket.error as e:
            print(f"Error sending TCP message: {e}")
            if not self.is_host:
                self.handle_disconnection("Lost connection to the host.")

    def broadcast_tcp(self, message_dict, exclude_ip=None):
        if not self.is_host or not self.running:
            return
            
        message_bytes = pickle.dumps(message_dict)
        for ip, conn in list(self.client_connections.items()):
            if ip != exclude_ip:
                try:
                    conn.sendall(message_bytes)
                except socket.error as e:
                    print(f"Failed to broadcast to {ip}: {e}. Removing client.")
                    if ip in self.client_connections:
                        del self.client_connections[ip]
                    if ip in self.clients:
                        del self.clients[ip]
                    self.root.after(0, self.remove_video_label, ip)
                    try:
                        conn.close()
                    except:
                        pass

    # --- UDP Handling & Media Streaming ---
    def listen_udp_video(self):
        while self.running:
            try:
                packet, addr = self.udp_video_socket.recvfrom(BUFFER_SIZE)
                ip = addr[0]
                if ip == self.my_ip:
                    continue
                    
                if ip in self.clients and self.clients[ip]["cam_on"]:
                    self.root.after(0, self.receive_video_frame, packet, ip)
            except socket.timeout:
                continue
            except socket.error as e:
                if self.running:
                    print(f"UDP Video receive error: {e}")
                break
            except Exception as e:
                if self.running:
                    print(f"Unexpected error in listen_udp_video: {e}")
                break
        print("UDP Video listening thread stopped.")

    def listen_udp_audio(self):
        while self.running:
            try:
                packet, addr = self.udp_audio_socket.recvfrom(BUFFER_SIZE)
                ip = addr[0]
                if ip == self.my_ip:
                    continue
                    
                if ip in self.clients and self.clients[ip]["mic_on"]:
                    self.audio_queue.put((packet, ip))
            except socket.timeout:
                continue
            except socket.error as e:
                if self.running:
                    print(f"UDP Audio receive error: {e}")
                break
            except Exception as e:
                if self.running:
                    print(f"Unexpected error in listen_udp_audio: {e}")
                break
        print("UDP Audio listening thread stopped.")

    def play_audio(self):
        while self.running:
            try:
                # Get audio data from queue with timeout
                packet, ip = self.audio_queue.get(timeout=1.0)
                if self.audio_stream_out and self.audio_stream_out.is_active():
                    self.audio_stream_out.write(packet)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error playing audio: {e}")
                time.sleep(0.1)
        print("Audio playback thread stopped.")

    def send_video_frames(self):
        while self.running and self.cam_enabled and self.video_capture and self.video_capture.isOpened():
            try:
                ret, frame = self.video_capture.read()
                if not ret:
                    time.sleep(0.1)
                    continue
                    
                self.root.after(0, self.update_self_video, frame)
                
                # Resize and compress frame
                frame_resized = cv2.resize(frame, (VIDEO_WIDTH, VIDEO_HEIGHT))
                ret, buffer = cv2.imencode(
                    ".jpg",
                    frame_resized,
                    [int(cv2.IMWRITE_JPEG_QUALITY), VIDEO_QUALITY]
                )
                
                if not ret:
                    continue
                    
                message = buffer.tobytes()
                if len(message) <= BUFFER_SIZE:
                    self.send_udp_packet(message, VIDEO_UDP_PORT)
                else:
                    print(f"Warning: Video frame size ({len(message)}) exceeds buffer size.")
                    
                time.sleep(1/30)  # ~30 FPS
            except Exception as e:
                print(f"Error in send_video_frames: {e}")
                time.sleep(1)
        print("Video sending thread stopped.")

    def send_audio_chunks(self):
        while self.running and self.mic_enabled and self.audio_stream_in and self.audio_stream_in.is_active():
            try:
                data = self.audio_stream_in.read(CHUNK, exception_on_overflow=False)
                self.send_udp_packet(data, AUDIO_UDP_PORT)
            except IOError as e:
                print(f"Audio read error: {e}")
                time.sleep(0.1)
            except Exception as e:
                print(f"Error in send_audio_chunks: {e}")
                time.sleep(1)
        print("Audio sending thread stopped.")

    def receive_video_frame(self, packet, ip):
        if not self.running:
            return
            
        try:
            # Skip if camera is off
            if not self.clients.get(ip, {}).get("cam_on", False):
                self.show_cam_off_placeholder(ip)
                return

            # Decode frame
            npdata = np.frombuffer(packet, dtype=np.uint8)
            frame = cv2.imdecode(npdata, cv2.IMREAD_COLOR)
            if frame is None:
                return

            # Update GUI
            label = self.video_labels.get(ip)
            if label:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                imgtk = ImageTk.PhotoImage(image=img)
                label.imgtk = imgtk
                label.config(image=imgtk)
        except Exception as e:
            pass  # Silently ignore frame processing errors

    def send_udp_packet(self, packet, port):
        if not self.running:
            return
            
        target_socket = self.udp_video_socket if port == VIDEO_UDP_PORT else self.udp_audio_socket
        if not target_socket:
            return

        # Get recipients
        recipients = []
        for ip in self.clients:
            if ip != self.my_ip:
                recipients.append(ip)

        # Send to each recipient
        for ip in set(recipients):
            try:
                target_socket.sendto(packet, (ip, port))
            except socket.error as e:
                pass  # Silently ignore send errors

    # --- GUI Updates & Chat ---
    def show_chat_screen(self):
        self.clear_frame()
        self.current_frame = tk.Frame(self.root)
        self.current_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # --- Top: Video Feeds Area ---
        self.video_area = tk.Frame(self.current_frame, bg="lightgrey")
        self.video_area.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.video_area.grid_rowconfigure(0, weight=1)
        self.video_area.grid_rowconfigure(1, weight=1)
        self.video_area.grid_columnconfigure(0, weight=1)
        self.video_area.grid_columnconfigure(1, weight=1)
        self.video_area.grid_columnconfigure(2, weight=1)

        self.add_self_video_label()
        self.update_client_video_labels()

        # --- Bottom Area (Controls and Chat) ---
        bottom_area = tk.Frame(self.current_frame, height=150)
        bottom_area.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)
        bottom_area.pack_propagate(False)

        # --- Bottom Left: Controls ---
        controls_frame = tk.Frame(bottom_area, width=200)
        controls_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        tk.Label(controls_frame, text=f"User: {self.username}", font=("Arial", 10)).pack(pady=2, anchor="w")
        role = "Host" if self.is_host else f"Client (Host: {self.host_ip})"
        tk.Label(controls_frame, text=role, font=("Arial", 10)).pack(pady=2, anchor="w")
        self.mic_button = tk.Button(controls_frame, text="Mute Mic", command=self.toggle_mic)
        self.mic_button.pack(pady=5, fill=tk.X)
        self.cam_button = tk.Button(controls_frame, text="Disable Cam", command=self.toggle_cam)
        self.cam_button.pack(pady=5, fill=tk.X)
        tk.Button(controls_frame, text="Exit Call", command=self.exit_call).pack(pady=5, fill=tk.X)

        # --- Bottom Right: Chat Box ---
        chat_frame = tk.Frame(bottom_area)
        chat_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        chat_scroll = Scrollbar(chat_frame)
        self.chat_display = Text(
            chat_frame,
            height=6,
            state="disabled",
            yscrollcommand=chat_scroll.set,
            wrap=tk.WORD,
            font=("Arial", 9)
        )
        chat_scroll.config(command=self.chat_display.yview)
        chat_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.chat_display.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        input_frame = tk.Frame(chat_frame)
        input_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self.chat_input = Entry(input_frame, font=("Arial", 9))
        self.chat_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.chat_input.bind("<Return>", self.send_chat_message_event)
        send_button = Button(input_frame, text="Send", command=self.send_chat_message)
        send_button.pack(side=tk.RIGHT)
        self.display_chat_message("--- Welcome to the chat! ---")

    def display_chat_message(self, message):
        if not self.chat_display or not self.running:
            return
            
        try:
            self.chat_display.config(state="normal")
            self.chat_display.insert(END, message + "\n")
            self.chat_display.config(state="disabled")
            self.chat_display.see(END)
        except tk.TclError:
            pass  # Silently ignore if widget is destroyed

    def send_chat_message_event(self, event):
        self.send_chat_message()

    def send_chat_message(self):
        if not self.running or not self.chat_input:
            return
            
        message_text = self.chat_input.get().strip()
        if message_text:
            self.chat_input.delete(0, END)
            self.display_chat_message(f"You: {message_text}")
            chat_msg = {
                "type": "chat",
                "username": self.username,
                "text": message_text
            }
            if self.is_host:
                self.broadcast_tcp(chat_msg)
            else:
                self.send_tcp_message(pickle.dumps(chat_msg))

    def add_self_video_label(self):
        if self.self_video_label:
            return
            
        frame_container = tk.Frame(self.video_area, bd=1, relief=tk.SOLID)
        video_label = Label(frame_container, bg="black")
        video_label.pack(fill=tk.BOTH, expand=True)
        name_label = Label(frame_container, text=f"{self.username} (You)", bg="grey", fg="white")
        name_label.pack(fill=tk.X)
        frame_container.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        self.self_video_label = video_label
        self.self_name_label = name_label
        self.update_self_video(None)
        self.update_status_indicator("self")

    def update_self_video(self, frame):
        if not self.running or not self.self_video_label:
            return
            
        try:
            if self.cam_enabled and frame is not None:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (VIDEO_WIDTH, VIDEO_HEIGHT))
                img = Image.fromarray(img)
                imgtk = ImageTk.PhotoImage(image=img)
                self.self_video_label.imgtk = imgtk
                self.self_video_label.config(image=imgtk)
            else:
                self.self_video_label.imgtk = self.cam_off_image
                self.self_video_label.config(image=self.cam_off_image)
        except Exception as e:
            print(f"Error updating self video: {e}")

    def update_client_video_labels(self):
        if not self.running:
            return
            
        for ip, client_data in self.clients.items():
            if ip == self.my_ip:
                continue
                
            if ip not in self.video_labels:
                self.add_video_label(
                    ip,
                    client_data["username"],
                    client_data["mic_on"],
                    client_data["cam_on"]
                )
            else:
                self.update_status_indicator(ip)
                if not client_data["cam_on"]:
                    self.show_cam_off_placeholder(ip)

    def add_video_label(self, ip, username, mic_on, cam_on):
        if not self.running or ip in self.video_labels or ip == self.my_ip:
            return
            
        print(f"Adding video label for {username} ({ip})")
        frame_container = tk.Frame(self.video_area, bd=1, relief=tk.SOLID)
        video_label = Label(frame_container, bg="black")
        video_label.pack(fill=tk.BOTH, expand=True)
        name_label = Label(frame_container, text=username, bg="grey", fg="white")
        name_label.pack(fill=tk.X)
        
        # Calculate grid position
        num_labels = len(self.video_labels) + 1  # +1 for self
        row = num_labels // 3
        col = num_labels % 3
        frame_container.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")
        
        self.video_labels[ip] = video_label
        self.name_labels[ip] = name_label
        
        # Store client if not already present
        if ip not in self.clients:
            self.clients[ip] = {
                "username": username,
                "mic_on": mic_on,
                "cam_on": cam_on
            }
            
        # Set initial state
        self.update_status_indicator(ip)
        if not cam_on:
            self.show_cam_off_placeholder(ip)
        else:
            # Show black until first frame arrives
            blank = Image.new("RGB", (VIDEO_WIDTH, VIDEO_HEIGHT), color="black")
            imgtk = ImageTk.PhotoImage(image=blank)
            video_label.imgtk = imgtk
            video_label.config(image=imgtk)

    def remove_video_label(self, ip):
        if not self.running or ip not in self.video_labels:
            return
            
        print(f"Removing video label for {ip}")
        label = self.video_labels.pop(ip)
        if ip in self.name_labels:
            del self.name_labels[ip]
            
        try:
            container = label.master
            container.destroy()
        except:
            pass
            
        self.regrid_video_labels()

    def regrid_video_labels(self):
        current_video_labels = list(self.video_labels.items())
        col_count = 3
        
        for i, (ip, label) in enumerate(current_video_labels):
            try:
                container = label.master
                row = (i + 1) // col_count  # +1 for self video
                col = (i + 1) % col_count
                container.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")
            except Exception as e:
                print(f"Error regridding label for {ip}: {e}")

    def update_status_indicator(self, ip):
        if not self.running:
            return
            
        if ip == "self":
            target_label = self.self_name_label
            mic_on = self.mic_enabled
            cam_on = self.cam_enabled
            username = f"{self.username} (You)"
        elif ip in self.clients:
            target_label = self.name_labels.get(ip)
            client_info = self.clients[ip]
            mic_on = client_info["mic_on"]
            cam_on = client_info["cam_on"]
            username = client_info["username"]
        else:
            return  # Client not found

        if not target_label:
            return

        status_text = username
        if not mic_on:
            status_text += " [Muted]"

        try:
            target_label.config(text=status_text)
        except tk.TclError:
            pass  # Silently ignore if widget is destroyed

    def show_cam_off_placeholder(self, ip):
        if not self.running:
            return
            
        label = self.video_labels.get(ip)
        if label:
            try:
                label.imgtk = self.cam_off_image
                label.config(image=self.cam_off_image)
            except tk.TclError:
                pass  # Silently ignore if widget is destroyed

    # --- Controls ---
    def toggle_mic(self):
        self.mic_enabled = not self.mic_enabled
        status_text = "Mute Mic" if self.mic_enabled else "Unmute Mic"
        self.mic_button.config(text=status_text)
        print(f"Mic {'enabled' if self.mic_enabled else 'disabled'}")
        
        # Update status indicator
        self.update_status_indicator("self")
        
        # Update host's client list
        if self.is_host and self.my_ip in self.clients:
            self.clients[self.my_ip]["mic_on"] = self.mic_enabled

        # Send status update
        msg = {
            "type": "mic_status",
            "username": self.username,
            "enabled": self.mic_enabled
        }
        if self.is_host:
            self.broadcast_tcp(msg)
        else:
            self.send_tcp_message(pickle.dumps(msg))

    def toggle_cam(self):
        self.cam_enabled = not self.cam_enabled
        status_text = "Disable Cam" if self.cam_enabled else "Enable Cam"
        self.cam_button.config(text=status_text)
        print(f"Camera {'enabled' if self.cam_enabled else 'disabled'}")
        
        # Update status indicator
        self.update_status_indicator("self")
        
        # Update host's client list
        if self.is_host and self.my_ip in self.clients:
            self.clients[self.my_ip]["cam_on"] = self.cam_enabled

        # Update self view
        self.root.after(0, self.update_self_video, None)
        
        # Send status update
        msg = {
            "type": "cam_status",
            "username": self.username,
            "enabled": self.cam_enabled
        }
        if self.is_host:
            self.broadcast_tcp(msg)
        else:
            self.send_tcp_message(pickle.dumps(msg))

    def exit_call(self):
        if messagebox.askyesno("Exit Call", "Are you sure you want to leave the chat?"):
            print("Exiting call...")
            self.running = False
            
            # Send leave notification
            if self.tcp_socket:
                if self.is_host:
                    self.broadcast_tcp({"type": "host_shutdown"})
                else:
                    try:
                        self.send_tcp_message(pickle.dumps({
                            "type": "leave",
                            "username": self.username
                        }))
                    except socket.error:
                        pass
            
            # Cleanup
            self.close_sockets()
            self.stop_media_streams()
            time.sleep(0.3)
            self.show_initial_screen()

    def handle_disconnection(self, message):
        if not self.running:
            return
            
        print(f"Handling disconnection: {message}")
        self.running = False
        self.close_sockets()
        self.stop_media_streams()
        self.root.after(0, lambda: messagebox.showinfo("Disconnected", message))
        self.root.after(100, self.show_initial_screen)

    def close_sockets(self):
        print("Closing sockets...")
        # Close client connections (host only)
        if self.is_host:
            for ip, conn in self.client_connections.items():
                try:
                    conn.shutdown(socket.SHUT_RDWR)
                except:
                    pass
                try:
                    conn.close()
                except:
                    pass
            self.client_connections = {}
        
        # Close main sockets
        if self.tcp_socket:
            try:
                self.tcp_socket.shutdown(socket.SHUT_RDWR)
            except:
                pass
            try:
                self.tcp_socket.close()
            except:
                pass
            self.tcp_socket = None
            
        if self.udp_video_socket:
            try:
                self.udp_video_socket.close()
            except:
                pass
            self.udp_video_socket = None
            
        if self.udp_audio_socket:
            try:
                self.udp_audio_socket.close()
            except:
                pass
            self.udp_audio_socket = None

    def on_closing(self):
        if self.running:
            if messagebox.askokcancel("Quit", "Do you want to exit the chat and quit?"):
                self.exit_call()
                self.root.after(100, self.root.destroy)
        else:
            self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoChatApp(root)
    root.mainloop()