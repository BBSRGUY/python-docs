# Python Operating System and System Programming

This document provides a comprehensive guide to operating system operations, system programming, and system administration in Python with syntax and usage examples.

## OS Module - Core Operating System Interface

### File and Directory Operations
```python
import os

# Current working directory
current_dir = os.getcwd()
print(f"Current directory: {current_dir}")

# Change directory
os.chdir('/tmp')  # Unix/Linux/macOS
# os.chdir('C:\\temp')  # Windows
print(f"Changed to: {os.getcwd()}")

# List directory contents
files = os.listdir('.')
print(f"Files in current directory: {files}")

# Create directories
os.mkdir('test_dir')                           # Create single directory
os.makedirs('nested/path/structure', exist_ok=True)  # Create nested directories

# Remove directories
os.rmdir('test_dir')                           # Remove empty directory
os.removedirs('nested/path/structure')         # Remove nested empty directories

# File operations
# Create a file
with open('test_file.txt', 'w') as f:
    f.write('Hello, World!')

# Check if path exists
print(f"File exists: {os.path.exists('test_file.txt')}")
print(f"Directory exists: {os.path.exists('/usr/bin')}")

# Check path type
print(f"Is file: {os.path.isfile('test_file.txt')}")
print(f"Is directory: {os.path.isdir('/usr/bin')}")
print(f"Is symlink: {os.path.islink('test_file.txt')}")

# File size and modification time
file_stats = os.stat('test_file.txt')
print(f"File size: {file_stats.st_size} bytes")
print(f"Modified time: {file_stats.st_mtime}")

# Remove file
os.remove('test_file.txt')

# Rename/move files
os.rename('old_name.txt', 'new_name.txt')

# File permissions (Unix/Linux/macOS)
if hasattr(os, 'chmod'):
    os.chmod('file.txt', 0o644)  # rw-r--r--
    os.chmod('script.py', 0o755)  # rwxr-xr-x
```

### Path Operations
```python
import os

# Path manipulation
path = '/home/user/documents/file.txt'

# Split path components
directory, filename = os.path.split(path)
print(f"Directory: {directory}")
print(f"Filename: {filename}")

# Split filename and extension
name, extension = os.path.splitext(filename)
print(f"Name: {name}")
print(f"Extension: {extension}")

# Get directory name
dirname = os.path.dirname(path)
print(f"Directory name: {dirname}")

# Get base name (filename)
basename = os.path.basename(path)
print(f"Base name: {basename}")

# Join paths (cross-platform)
joined_path = os.path.join('home', 'user', 'documents', 'file.txt')
print(f"Joined path: {joined_path}")

# Absolute path
abs_path = os.path.abspath('relative/path/file.txt')
print(f"Absolute path: {abs_path}")

# Real path (resolves symlinks)
real_path = os.path.realpath('symlink_file.txt')
print(f"Real path: {real_path}")

# Relative path
rel_path = os.path.relpath('/home/user/docs/file.txt', '/home/user')
print(f"Relative path: {rel_path}")

# Common path operations
paths = ['/home/user/docs', '/home/user/pictures', '/home/user/music']
common_prefix = os.path.commonprefix(paths)
print(f"Common prefix: {common_prefix}")

# Platform-specific path separator
print(f"Path separator: '{os.sep}'")
print(f"Alt separator: '{os.altsep}'")
print(f"Path list separator: '{os.pathsep}'")
```

### Environment Variables
```python
import os

# Get environment variable
home_dir = os.environ.get('HOME')  # Unix/Linux/macOS
# home_dir = os.environ.get('USERPROFILE')  # Windows
print(f"Home directory: {home_dir}")

# Get with default value
python_path = os.environ.get('PYTHONPATH', 'Not set')
print(f"PYTHON PATH: {python_path}")

# Set environment variable
os.environ['MY_VAR'] = 'my_value'
print(f"MY_VAR: {os.environ['MY_VAR']}")

# Get all environment variables
print("All environment variables:")
for key, value in os.environ.items():
    print(f"{key}: {value}")

# Expand environment variables in paths
expanded_path = os.path.expandvars('$HOME/documents')  # Unix/Linux/macOS
# expanded_path = os.path.expandvars('%USERPROFILE%\\Documents')  # Windows
print(f"Expanded path: {expanded_path}")

# Expand user home directory
user_path = os.path.expanduser('~/documents')
print(f"User path: {user_path}")

# Platform-specific variables
if os.name == 'posix':  # Unix/Linux/macOS
    print("Running on Unix-like system")
    shell = os.environ.get('SHELL', 'Unknown')
    print(f"Shell: {shell}")
elif os.name == 'nt':  # Windows
    print("Running on Windows")
    comspec = os.environ.get('COMSPEC', 'Unknown')
    print(f"Command processor: {comspec}")
```

### Process Management
```python
import os
import sys

# Current process information
print(f"Process ID: {os.getpid()}")
print(f"Parent Process ID: {os.getppid()}")

# User and group information (Unix/Linux/macOS)
if hasattr(os, 'getuid'):
    print(f"User ID: {os.getuid()}")
    print(f"Group ID: {os.getgid()}")
    print(f"Effective User ID: {os.geteuid()}")
    print(f"Effective Group ID: {os.getegid()}")

# Execute system commands
# Method 1: os.system() - simple but limited
return_code = os.system('ls -la')  # Unix/Linux/macOS
# return_code = os.system('dir')   # Windows
print(f"Command return code: {return_code}")

# Method 2: os.popen() - capture output
with os.popen('date') as f:  # Unix/Linux/macOS
    output = f.read()
    print(f"Command output: {output.strip()}")

# Method 3: os.exec*() family - replace current process
# os.execv('/bin/ls', ['ls', '-la'])  # This would replace current process

# Method 4: os.spawn*() family - spawn new process
if hasattr(os, 'spawnv'):
    pid = os.spawnv(os.P_NOWAIT, '/bin/echo', ['echo', 'Hello from spawn'])
    print(f"Spawned process PID: {pid}")

# Exit the program
# sys.exit(0)  # Normal exit
# sys.exit(1)  # Exit with error code
```

## Subprocess Module - Advanced Process Management

### Running External Commands
```python
import subprocess
import sys
import time

# Basic command execution
result = subprocess.run(['ls', '-la'], capture_output=True, text=True)
print(f"Return code: {result.returncode}")
print(f"STDOUT:\n{result.stdout}")
print(f"STDERR:\n{result.stderr}")

# Windows equivalent
if sys.platform == 'win32':
    result = subprocess.run(['dir'], shell=True, capture_output=True, text=True)

# Run with timeout
try:
    result = subprocess.run(['sleep', '2'], timeout=1)
except subprocess.TimeoutExpired:
    print("Command timed out")

# Check if command succeeded
try:
    result = subprocess.run(['ls', '/nonexistent'], check=True, capture_output=True)
except subprocess.CalledProcessError as e:
    print(f"Command failed with return code {e.returncode}")
    print(f"Error output: {e.stderr}")

# Pipe commands together
p1 = subprocess.Popen(['ls', '-la'], stdout=subprocess.PIPE)
p2 = subprocess.Popen(['grep', 'py'], stdin=p1.stdout, stdout=subprocess.PIPE)
p1.stdout.close()  # Allow p1 to receive SIGPIPE if p2 exits
output, _ = p2.communicate()
print(f"Piped output: {output.decode()}")

# Interactive process
class InteractiveProcess:
    def __init__(self, command):
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True
        )
    
    def send_command(self, command):
        """Send command to process"""
        self.process.stdin.write(f"{command}\n")
        self.process.stdin.flush()
    
    def read_output(self, timeout=1):
        """Read output with timeout"""
        try:
            output, error = self.process.communicate(timeout=timeout)
            return output, error
        except subprocess.TimeoutExpired:
            return None, None
    
    def is_running(self):
        """Check if process is still running"""
        return self.process.poll() is None
    
    def terminate(self):
        """Terminate the process"""
        self.process.terminate()
        self.process.wait()

# Example usage of interactive process
# proc = InteractiveProcess(['python3', '-i'])
# proc.send_command('print("Hello from Python")')
# output, error = proc.read_output()
# proc.terminate()
```

### Advanced Subprocess Features
```python
import subprocess
import os
import signal
import threading

# Environment manipulation
env = os.environ.copy()
env['MY_CUSTOM_VAR'] = 'custom_value'

result = subprocess.run(
    ['python3', '-c', 'import os; print(os.environ.get("MY_CUSTOM_VAR"))'],
    env=env,
    capture_output=True,
    text=True
)
print(f"Custom environment result: {result.stdout.strip()}")

# Working directory
result = subprocess.run(
    ['pwd'],  # Unix/Linux/macOS
    cwd='/tmp',
    capture_output=True,
    text=True
)
print(f"Working directory: {result.stdout.strip()}")

# Input to process
result = subprocess.run(
    ['python3', '-c', 'name = input("Enter name: "); print(f"Hello, {name}")'],
    input='Alice\n',
    capture_output=True,
    text=True
)
print(f"Input result: {result.stdout}")

# Real-time output streaming
def stream_command(command):
    """Stream command output in real-time"""
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(f"Stream: {output.strip()}")
    
    return process.poll()

# Example: stream a long-running command
# return_code = stream_command(['ping', '-c', '5', 'google.com'])

# Process groups and signal handling
def run_process_group():
    """Run process in new process group"""
    process = subprocess.Popen(
        ['sleep', '10'],
        preexec_fn=os.setsid  # Create new process group
    )
    
    # Kill entire process group
    try:
        time.sleep(1)
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except ProcessLookupError:
        pass
    
    return process.wait()

# Parallel process execution
def run_parallel_processes(commands):
    """Run multiple commands in parallel"""
    processes = []
    
    for cmd in commands:
        process = subprocess.Popen(cmd, capture_output=True, text=True)
        processes.append(process)
    
    # Wait for all processes to complete
    results = []
    for process in processes:
        stdout, stderr = process.communicate()
        results.append({
            'returncode': process.returncode,
            'stdout': stdout,
            'stderr': stderr
        })
    
    return results

# Example parallel execution
commands = [
    ['echo', 'Process 1'],
    ['echo', 'Process 2'],
    ['echo', 'Process 3']
]
# results = run_parallel_processes(commands)
```

## Platform Module - Platform Information

### System Information
```python
import platform
import sys

# Basic platform information
print(f"System: {platform.system()}")           # Linux, Windows, Darwin (macOS)
print(f"Node: {platform.node()}")               # Computer name
print(f"Release: {platform.release()}")         # OS release
print(f"Version: {platform.version()}")         # OS version
print(f"Machine: {platform.machine()}")         # Hardware type
print(f"Processor: {platform.processor()}")     # Processor type
print(f"Architecture: {platform.architecture()}")  # Architecture info

# Detailed platform information
print(f"Platform: {platform.platform()}")       # Complete platform string
print(f"Detailed platform: {platform.platform(aliased=True, terse=False)}")

# Python information
print(f"Python version: {platform.python_version()}")
print(f"Python implementation: {platform.python_implementation()}")
print(f"Python compiler: {platform.python_compiler()}")
print(f"Python build: {platform.python_build()}")

# System-specific information
if platform.system() == 'Linux':
    try:
        dist_info = platform.freedesktop_os_release()
        print(f"Linux distribution: {dist_info}")
    except:
        print("Could not determine Linux distribution")

elif platform.system() == 'Windows':
    win_ver = platform.win32_ver()
    print(f"Windows version: {win_ver}")
    
    # Additional Windows info
    win_edition = platform.win32_edition()
    print(f"Windows edition: {win_edition}")

elif platform.system() == 'Darwin':  # macOS
    mac_ver = platform.mac_ver()
    print(f"macOS version: {mac_ver}")

# Java information (if available)
try:
    java_ver = platform.java_ver()
    print(f"Java version: {java_ver}")
except:
    print("Java not available")

# Libc information (Unix/Linux)
try:
    libc_ver = platform.libc_ver()
    print(f"Libc version: {libc_ver}")
except:
    print("Libc information not available")

# System capabilities
print(f"sys.platform: {sys.platform}")
print(f"os.name: {os.name}")
print(f"sys.maxsize: {sys.maxsize}")
print(f"sys.byteorder: {sys.byteorder}")
```

### Hardware Information
```python
import platform
import os
import subprocess
import psutil  # pip install psutil

class SystemInfo:
    @staticmethod
    def get_cpu_info():
        """Get CPU information"""
        info = {
            'processor': platform.processor(),
            'machine': platform.machine(),
            'architecture': platform.architecture()[0]
        }
        
        # Try to get more detailed CPU info
        if platform.system() == 'Linux':
            try:
                with open('/proc/cpuinfo', 'r') as f:
                    cpuinfo = f.read()
                    # Parse CPU info
                    for line in cpuinfo.split('\n'):
                        if 'model name' in line:
                            info['model_name'] = line.split(':')[1].strip()
                            break
            except:
                pass
        
        elif platform.system() == 'Windows':
            try:
                result = subprocess.run(
                    ['wmic', 'cpu', 'get', 'name'],
                    capture_output=True, text=True
                )
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:
                    info['model_name'] = lines[1].strip()
            except:
                pass
        
        elif platform.system() == 'Darwin':  # macOS
            try:
                result = subprocess.run(
                    ['sysctl', '-n', 'machdep.cpu.brand_string'],
                    capture_output=True, text=True
                )
                info['model_name'] = result.stdout.strip()
            except:
                pass
        
        return info
    
    @staticmethod
    def get_memory_info():
        """Get memory information"""
        info = {}
        
        if platform.system() == 'Linux':
            try:
                with open('/proc/meminfo', 'r') as f:
                    meminfo = f.read()
                    for line in meminfo.split('\n'):
                        if 'MemTotal:' in line:
                            total_kb = int(line.split()[1])
                            info['total_memory'] = total_kb * 1024  # Convert to bytes
                        elif 'MemAvailable:' in line:
                            available_kb = int(line.split()[1])
                            info['available_memory'] = available_kb * 1024
            except:
                pass
        
        elif platform.system() == 'Windows':
            try:
                result = subprocess.run(
                    ['wmic', 'computersystem', 'get', 'TotalPhysicalMemory'],
                    capture_output=True, text=True
                )
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:
                    info['total_memory'] = int(lines[1].strip())
            except:
                pass
        
        elif platform.system() == 'Darwin':  # macOS
            try:
                result = subprocess.run(
                    ['sysctl', '-n', 'hw.memsize'],
                    capture_output=True, text=True
                )
                info['total_memory'] = int(result.stdout.strip())
            except:
                pass
        
        return info
    
    @staticmethod
    def get_disk_info():
        """Get disk information"""
        info = []
        
        if platform.system() == 'Linux':
            try:
                result = subprocess.run(
                    ['df', '-h'], capture_output=True, text=True
                )
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                for line in lines:
                    parts = line.split()
                    if len(parts) >= 6:
                        info.append({
                            'filesystem': parts[0],
                            'size': parts[1],
                            'used': parts[2],
                            'available': parts[3],
                            'use_percent': parts[4],
                            'mount_point': parts[5]
                        })
            except:
                pass
        
        return info

# Example usage
sys_info = SystemInfo()
cpu_info = sys_info.get_cpu_info()
memory_info = sys_info.get_memory_info()
disk_info = sys_info.get_disk_info()

print("CPU Information:")
for key, value in cpu_info.items():
    print(f"  {key}: {value}")

print("\nMemory Information:")
for key, value in memory_info.items():
    if 'memory' in key:
        print(f"  {key}: {value / (1024**3):.2f} GB")
    else:
        print(f"  {key}: {value}")
```

## Pathlib Module - Object-Oriented Path Handling

### Path Objects and Operations
```python
from pathlib import Path, PurePath
import os

# Create Path objects
current_path = Path.cwd()
home_path = Path.home()
file_path = Path('documents/file.txt')
absolute_path = Path('/usr/local/bin/python')

print(f"Current directory: {current_path}")
print(f"Home directory: {home_path}")

# Path properties
print(f"File name: {file_path.name}")           # file.txt
print(f"File stem: {file_path.stem}")           # file
print(f"File suffix: {file_path.suffix}")       # .txt
print(f"File suffixes: {file_path.suffixes}")   # ['.txt']
print(f"Parent directory: {file_path.parent}")  # documents
print(f"All parents: {list(file_path.parents)}")  # [documents, .]

# Path manipulation
new_path = file_path.with_suffix('.md')         # documents/file.md
new_name = file_path.with_name('newfile.txt')   # documents/newfile.txt
new_stem = file_path.with_stem('newfile')       # documents/newfile.txt

# Joining paths
joined_path = current_path / 'documents' / 'file.txt'
print(f"Joined path: {joined_path}")

# Path resolution
relative_path = Path('../documents/file.txt')
resolved_path = relative_path.resolve()
print(f"Resolved path: {resolved_path}")

# Check path properties
print(f"Path exists: {current_path.exists()}")
print(f"Is file: {file_path.is_file()}")
print(f"Is directory: {current_path.is_dir()}")
print(f"Is symlink: {file_path.is_symlink()}")
print(f"Is absolute: {absolute_path.is_absolute()}")

# File operations with pathlib
test_file = Path('test.txt')

# Create file
test_file.write_text('Hello, World!')
print(f"File created: {test_file.exists()}")

# Read file
content = test_file.read_text()
print(f"File content: {content}")

# File statistics
if test_file.exists():
    stat = test_file.stat()
    print(f"File size: {stat.st_size} bytes")
    print(f"Modified time: {stat.st_mtime}")
    print(f"File mode: {oct(stat.st_mode)}")

# Binary operations
test_file.write_bytes(b'Binary data')
binary_content = test_file.read_bytes()
print(f"Binary content: {binary_content}")

# Directory operations
test_dir = Path('test_directory')
test_dir.mkdir(exist_ok=True)

# Create nested directories
nested_dir = Path('nested/deep/structure')
nested_dir.mkdir(parents=True, exist_ok=True)

# Cleanup
test_file.unlink()  # Remove file
test_dir.rmdir()    # Remove empty directory
```

### Path Iteration and Globbing
```python
from pathlib import Path
import fnmatch

# Iterate over directory contents
current_dir = Path('.')

# List all items
for item in current_dir.iterdir():
    if item.is_file():
        print(f"File: {item}")
    elif item.is_dir():
        print(f"Directory: {item}")

# Glob patterns
# Find all Python files
py_files = list(current_dir.glob('*.py'))
print(f"Python files: {py_files}")

# Recursive glob
all_py_files = list(current_dir.rglob('*.py'))
print(f"All Python files (recursive): {all_py_files}")

# Complex glob patterns
text_files = list(current_dir.glob('**/*.txt'))
config_files = list(current_dir.glob('**/config.*'))

# Custom filtering
def find_large_files(directory, min_size_mb=10):
    """Find files larger than min_size_mb"""
    large_files = []
    for file_path in directory.rglob('*'):
        if file_path.is_file():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            if size_mb > min_size_mb:
                large_files.append((file_path, size_mb))
    return large_files

# Find files by pattern and date
import datetime

def find_recent_files(directory, pattern='*', days=7):
    """Find files modified within the last N days"""
    cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days)
    recent_files = []
    
    for file_path in directory.rglob(pattern):
        if file_path.is_file():
            mod_time = datetime.datetime.fromtimestamp(file_path.stat().st_mtime)
            if mod_time > cutoff_date:
                recent_files.append((file_path, mod_time))
    
    return sorted(recent_files, key=lambda x: x[1], reverse=True)

# Example usage
# large_files = find_large_files(Path('.'), min_size_mb=1)
# recent_files = find_recent_files(Path('.'), pattern='*.py', days=30)

# Advanced path operations
class PathUtilities:
    @staticmethod
    def copy_file(src, dst):
        """Copy file using pathlib"""
        src_path = Path(src)
        dst_path = Path(dst)
        
        # Ensure destination directory exists
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy file
        dst_path.write_bytes(src_path.read_bytes())
        return dst_path
    
    @staticmethod
    def move_file(src, dst):
        """Move file using pathlib"""
        src_path = Path(src)
        dst_path = Path(dst)
        
        # Ensure destination directory exists
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Move file
        src_path.rename(dst_path)
        return dst_path
    
    @staticmethod
    def get_directory_size(directory):
        """Calculate total size of directory"""
        total_size = 0
        for file_path in Path(directory).rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size
    
    @staticmethod
    def find_duplicates(directory):
        """Find duplicate files by content hash"""
        import hashlib
        
        hashes = {}
        duplicates = []
        
        for file_path in Path(directory).rglob('*'):
            if file_path.is_file():
                # Calculate file hash
                hash_md5 = hashlib.md5()
                with open(file_path, 'rb') as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        hash_md5.update(chunk)
                
                file_hash = hash_md5.hexdigest()
                
                if file_hash in hashes:
                    duplicates.append((hashes[file_hash], file_path))
                else:
                    hashes[file_hash] = file_path
        
        return duplicates
```

## Shutil Module - High-Level File Operations

### File and Directory Operations
```python
import shutil
import os
from pathlib import Path

# Copy operations
# Copy file
shutil.copy('source.txt', 'destination.txt')               # Copy file
shutil.copy2('source.txt', 'dest_with_metadata.txt')       # Copy with metadata
shutil.copyfile('source.txt', 'dest_content_only.txt')     # Copy content only
shutil.copymode('source.txt', 'dest.txt')                  # Copy permissions only
shutil.copystat('source.txt', 'dest.txt')                  # Copy stat info

# Copy directory tree
shutil.copytree('source_dir', 'destination_dir')
shutil.copytree('source_dir', 'dest_dir', dirs_exist_ok=True)  # Allow existing dest

# Custom copy function
def custom_copy_function(src, dst):
    """Custom copy with filtering"""
    def ignore_patterns(dir, files):
        return [f for f in files if f.endswith('.tmp') or f.startswith('.')]
    
    shutil.copytree(src, dst, ignore=ignore_patterns)

# Move operations
shutil.move('source.txt', 'new_location.txt')              # Move file
shutil.move('source_dir', 'new_location_dir')              # Move directory

# Remove operations
shutil.rmtree('directory_to_remove')                       # Remove directory tree
shutil.rmtree('dir', ignore_errors=True)                   # Ignore errors

# Disk usage
total, used, free = shutil.disk_usage('/')                 # Unix/Linux/macOS
# total, used, free = shutil.disk_usage('C:\\')            # Windows
print(f"Total: {total // (2**30)} GB")
print(f"Used: {used // (2**30)} GB")
print(f"Free: {free // (2**30)} GB")

# Archive operations
# Create archive
shutil.make_archive('backup', 'zip', 'directory_to_archive')
shutil.make_archive('backup', 'tar', 'directory_to_archive')
shutil.make_archive('backup', 'gztar', 'directory_to_archive')  # .tar.gz

# Extract archive
shutil.unpack_archive('backup.zip', 'extract_to_directory')
shutil.unpack_archive('backup.tar.gz', 'extract_directory')

# Get available archive formats
formats = shutil.get_archive_formats()
print(f"Available archive formats: {formats}")

# Which command (find executable)
python_path = shutil.which('python3')
print(f"Python3 path: {python_path}")

git_path = shutil.which('git')
print(f"Git path: {git_path}")
```

### Advanced File Operations
```python
import shutil
import os
import stat
import time
import hashlib
from pathlib import Path

class AdvancedFileOperations:
    @staticmethod
    def secure_delete(file_path, passes=3):
        """Securely delete file by overwriting"""
        path = Path(file_path)
        if not path.exists():
            return False
        
        file_size = path.stat().st_size
        
        with open(path, 'r+b') as f:
            for _ in range(passes):
                f.seek(0)
                f.write(os.urandom(file_size))
                f.flush()
                os.fsync(f.fileno())  # Force write to disk
        
        path.unlink()
        return True
    
    @staticmethod
    def backup_with_rotation(source, backup_dir, max_backups=5):
        """Create backup with rotation"""
        backup_path = Path(backup_dir)
        backup_path.mkdir(parents=True, exist_ok=True)
        
        source_path = Path(source)
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        backup_name = f"{source_path.stem}_{timestamp}{source_path.suffix}"
        
        # Create backup
        backup_file = backup_path / backup_name
        shutil.copy2(source, backup_file)
        
        # Remove old backups
        backup_files = sorted(backup_path.glob(f"{source_path.stem}_*{source_path.suffix}"))
        if len(backup_files) > max_backups:
            for old_backup in backup_files[:-max_backups]:
                old_backup.unlink()
        
        return backup_file
    
    @staticmethod
    def sync_directories(src, dst, delete_extra=False):
        """Synchronize two directories"""
        src_path = Path(src)
        dst_path = Path(dst)
        
        dst_path.mkdir(parents=True, exist_ok=True)
        
        # Copy new and updated files
        for src_file in src_path.rglob('*'):
            if src_file.is_file():
                rel_path = src_file.relative_to(src_path)
                dst_file = dst_path / rel_path
                
                # Create parent directories
                dst_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Copy if file doesn't exist or is newer
                if not dst_file.exists() or src_file.stat().st_mtime > dst_file.stat().st_mtime:
                    shutil.copy2(src_file, dst_file)
                    print(f"Copied: {rel_path}")
        
        # Remove extra files if requested
        if delete_extra:
            for dst_file in dst_path.rglob('*'):
                if dst_file.is_file():
                    rel_path = dst_file.relative_to(dst_path)
                    src_file = src_path / rel_path
                    
                    if not src_file.exists():
                        dst_file.unlink()
                        print(f"Removed: {rel_path}")
    
    @staticmethod
    def verify_copy(src, dst):
        """Verify file copy by comparing checksums"""
        def get_file_hash(file_path):
            hash_md5 = hashlib.md5()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        
        src_hash = get_file_hash(src)
        dst_hash = get_file_hash(dst)
        
        return src_hash == dst_hash
    
    @staticmethod
    def change_permissions_recursive(directory, file_mode=0o644, dir_mode=0o755):
        """Change permissions recursively"""
        for root, dirs, files in os.walk(directory):
            # Set directory permissions
            for d in dirs:
                dir_path = os.path.join(root, d)
                os.chmod(dir_path, dir_mode)
            
            # Set file permissions
            for f in files:
                file_path = os.path.join(root, f)
                os.chmod(file_path, file_mode)

# Example usage
file_ops = AdvancedFileOperations()

# Create test file
test_file = Path('test_secure.txt')
test_file.write_text('Sensitive data')

# Secure delete
# file_ops.secure_delete(test_file)

# Backup with rotation
# backup_file = file_ops.backup_with_rotation('important.txt', 'backups')

# Verify copy
# shutil.copy2('source.txt', 'destination.txt')
# is_verified = file_ops.verify_copy('source.txt', 'destination.txt')
```

## Tempfile Module - Temporary Files and Directories

### Temporary File Operations
```python
import tempfile
import os
from pathlib import Path

# Temporary files
# Method 1: NamedTemporaryFile
with tempfile.NamedTemporaryFile(mode='w+t', delete=True, suffix='.txt') as temp_file:
    temp_file.write('Temporary content')
    temp_file.seek(0)
    content = temp_file.read()
    print(f"Temp file name: {temp_file.name}")
    print(f"Temp content: {content}")
# File is automatically deleted when context exits

# Method 2: Temporary file that persists
temp_file = tempfile.NamedTemporaryFile(mode='w+t', delete=False, suffix='.log')
temp_filename = temp_file.name
temp_file.write('Persistent temporary file')
temp_file.close()

# Use the file
with open(temp_filename, 'r') as f:
    content = f.read()
    print(f"Persistent temp content: {content}")

# Manually delete
os.unlink(temp_filename)

# Method 3: Get temporary file descriptor
fd, temp_path = tempfile.mkstemp(suffix='.tmp', prefix='myapp_')
try:
    with os.fdopen(fd, 'w') as temp_file:
        temp_file.write('Content via file descriptor')
    
    # Use the file
    with open(temp_path, 'r') as f:
        content = f.read()
        print(f"FD temp content: {content}")
finally:
    os.unlink(temp_path)

# Temporary directories
# Method 1: TemporaryDirectory context manager
with tempfile.TemporaryDirectory() as temp_dir:
    print(f"Temp directory: {temp_dir}")
    
    # Create files in temp directory
    temp_file_path = Path(temp_dir) / 'temp_file.txt'
    temp_file_path.write_text('File in temp directory')
    
    # List contents
    for item in Path(temp_dir).iterdir():
        print(f"Temp item: {item}")
# Directory is automatically deleted

# Method 2: Manual temporary directory
temp_dir = tempfile.mkdtemp(prefix='myapp_', suffix='_temp')
print(f"Manual temp dir: {temp_dir}")

try:
    # Use the directory
    temp_file = Path(temp_dir) / 'file.txt'
    temp_file.write_text('Manual temp directory file')
finally:
    # Manual cleanup
    import shutil
    shutil.rmtree(temp_dir)

# Configure temporary file location
# Get default temp directory
default_temp = tempfile.gettempdir()
print(f"Default temp directory: {default_temp}")

# Get user-specific temp directory
user_temp = tempfile.gettempdirb()
print(f"User temp directory: {user_temp}")

# Set custom temp directory
original_tempdir = tempfile.tempdir
tempfile.tempdir = '/tmp/custom'  # Unix/Linux/macOS
# tempfile.tempdir = 'C:\\Temp\\Custom'  # Windows

# Create temp file in custom location
with tempfile.NamedTemporaryFile() as temp_file:
    print(f"Custom temp file: {temp_file.name}")

# Restore original temp directory
tempfile.tempdir = original_tempdir
```

### Secure Temporary Files
```python
import tempfile
import os
import stat
from pathlib import Path

class SecureTempOperations:
    @staticmethod
    def create_secure_temp_file(content, mode=0o600):
        """Create temporary file with secure permissions"""
        fd, temp_path = tempfile.mkstemp()
        
        try:
            # Set secure permissions
            os.fchmod(fd, mode)
            
            # Write content
            with os.fdopen(fd, 'w') as temp_file:
                temp_file.write(content)
            
            return temp_path
        except:
            # Clean up on error
            try:
                os.unlink(temp_path)
            except:
                pass
            raise
    
    @staticmethod
    def create_secure_temp_dir(mode=0o700):
        """Create temporary directory with secure permissions"""
        temp_dir = tempfile.mkdtemp()
        
        # Set secure permissions
        os.chmod(temp_dir, mode)
        
        return temp_dir
    
    @staticmethod
    def atomic_write(target_file, content):
        """Atomically write to file using temporary file"""
        target_path = Path(target_file)
        temp_dir = target_path.parent
        
        # Create temporary file in same directory
        with tempfile.NamedTemporaryFile(
            mode='w',
            dir=temp_dir,
            delete=False,
            prefix=f'.{target_path.name}_tmp_'
        ) as temp_file:
            temp_file.write(content)
            temp_name = temp_file.name
        
        try:
            # Atomic rename
            os.rename(temp_name, target_file)
        except:
            # Clean up on error
            try:
                os.unlink(temp_name)
            except:
                pass
            raise
    
    @staticmethod
    def create_temp_workspace():
        """Create a temporary workspace with multiple files"""
        workspace = tempfile.mkdtemp(prefix='workspace_')
        
        workspace_path = Path(workspace)
        
        # Create subdirectories
        (workspace_path / 'input').mkdir()
        (workspace_path / 'output').mkdir()
        (workspace_path / 'temp').mkdir()
        
        # Create configuration file
        config_file = workspace_path / 'config.ini'
        config_file.write_text('''[settings]
debug = true
temp_dir = temp
output_dir = output
''')
        
        return workspace

# Example usage
secure_ops = SecureTempOperations()

# Secure temporary file
secure_temp = secure_ops.create_secure_temp_file('Sensitive data', mode=0o600)
print(f"Secure temp file: {secure_temp}")
# Clean up
os.unlink(secure_temp)

# Secure temporary directory
secure_dir = secure_ops.create_secure_temp_dir(mode=0o700)
print(f"Secure temp dir: {secure_dir}")
# Clean up
import shutil
shutil.rmtree(secure_dir)

# Atomic write
secure_ops.atomic_write('important_file.txt', 'Critical data')

# Temporary workspace
workspace = secure_ops.create_temp_workspace()
print(f"Workspace created: {workspace}")
# List workspace contents
for item in Path(workspace).rglob('*'):
    print(f"Workspace item: {item}")
# Clean up
shutil.rmtree(workspace)
```

## Glob Module - Unix-Style Pathname Pattern Matching

### Pattern Matching
```python
import glob
import os
from pathlib import Path

# Basic glob patterns
print("Python files:")
py_files = glob.glob('*.py')
for file in py_files:
    print(f"  {file}")

# Recursive glob (Python 3.5+)
print("\nAll Python files (recursive):")
all_py_files = glob.glob('**/*.py', recursive=True)
for file in all_py_files:
    print(f"  {file}")

# iglob for iterator (memory efficient)
print("\nPython files (iterator):")
for file in glob.iglob('*.py'):
    print(f"  {file}")

# Different patterns
text_files = glob.glob('*.txt')                 # All .txt files
config_files = glob.glob('*config*')            # Files containing 'config'
hidden_files = glob.glob('.*')                  # Hidden files (Unix/Linux/macOS)

# Character classes
single_digit = glob.glob('file[0-9].txt')       # file0.txt, file1.txt, etc.
vowel_files = glob.glob('*[aeiou]*.txt')        # Files with vowels
not_backup = glob.glob('*[!~]')                 # Files not ending with ~

# Escape special characters
literal_bracket = glob.glob('file[[]1].txt')    # Literal bracket: file[1].txt

# Multiple patterns
def multi_glob(*patterns):
    """Combine multiple glob patterns"""
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern))
    return list(set(files))  # Remove duplicates

# Example
source_files = multi_glob('*.py', '*.c', '*.cpp', '*.java')
print(f"\nSource files: {source_files}")

# Advanced patterns
class AdvancedGlob:
    @staticmethod
    def find_by_size(pattern, min_size=0, max_size=float('inf')):
        """Find files by pattern and size"""
        matching_files = []
        for file_path in glob.glob(pattern):
            if os.path.isfile(file_path):
                size = os.path.getsize(file_path)
                if min_size <= size <= max_size:
                    matching_files.append((file_path, size))
        return matching_files
    
    @staticmethod
    def find_by_date(pattern, days_old=None, newer_than=None):
        """Find files by pattern and modification date"""
        import time
        from datetime import datetime, timedelta
        
        matching_files = []
        current_time = time.time()
        
        for file_path in glob.glob(pattern):
            if os.path.isfile(file_path):
                mod_time = os.path.getmtime(file_path)
                
                if days_old is not None:
                    cutoff_time = current_time - (days_old * 24 * 60 * 60)
                    if mod_time < cutoff_time:
                        matching_files.append(file_path)
                
                elif newer_than is not None:
                    if mod_time > newer_than:
                        matching_files.append(file_path)
        
        return matching_files
    
    @staticmethod
    def find_duplicates_by_name(pattern):
        """Find duplicate file names (different paths)"""
        files = glob.glob(pattern, recursive=True)
        name_groups = {}
        
        for file_path in files:
            name = os.path.basename(file_path)
            if name not in name_groups:
                name_groups[name] = []
            name_groups[name].append(file_path)
        
        # Return only groups with duplicates
        duplicates = {name: paths for name, paths in name_groups.items() if len(paths) > 1}
        return duplicates

# Example usage
advanced_glob = AdvancedGlob()

# Find large files
large_files = advanced_glob.find_by_size('**/*', min_size=1024*1024, max_size=10*1024*1024)  # 1MB to 10MB
print(f"\nLarge files (1MB-10MB): {len(large_files)}")

# Find recent files
import time
one_week_ago = time.time() - (7 * 24 * 60 * 60)
recent_files = advanced_glob.find_by_date('**/*', newer_than=one_week_ago)
print(f"Recent files (last week): {len(recent_files)}")

# Find duplicate names
duplicate_names = advanced_glob.find_duplicates_by_name('**/*')
print(f"Duplicate file names: {len(duplicate_names)}")
```

### Cross-Platform Globbing
```python
import glob
import fnmatch
import os
from pathlib import Path

class CrossPlatformGlob:
    @staticmethod
    def safe_glob(pattern, case_sensitive=None):
        """Cross-platform glob with case sensitivity control"""
        if case_sensitive is None:
            # Default case sensitivity based on platform
            case_sensitive = os.name != 'nt'  # Case sensitive except on Windows
        
        if case_sensitive:
            return glob.glob(pattern)
        else:
            # Case insensitive matching
            all_files = glob.glob('**/*', recursive=True) if '**' in pattern else glob.glob('*')
            pattern_lower = pattern.lower()
            return [f for f in all_files if fnmatch.fnmatch(f.lower(), pattern_lower)]
    
    @staticmethod
    def normalize_path_pattern(pattern):
        """Normalize path separators in glob pattern"""
        # Convert to platform-specific separators
        return pattern.replace('/', os.sep).replace('\\', os.sep)
    
    @staticmethod
    def glob_with_exclusions(include_pattern, exclude_patterns=None):
        """Glob with exclusion patterns"""
        if exclude_patterns is None:
            exclude_patterns = []
        
        included_files = glob.glob(include_pattern, recursive=True)
        
        # Filter out excluded files
        filtered_files = []
        for file_path in included_files:
            excluded = False
            for exclude_pattern in exclude_patterns:
                if fnmatch.fnmatch(file_path, exclude_pattern):
                    excluded = True
                    break
            if not excluded:
                filtered_files.append(file_path)
        
        return filtered_files
    
    @staticmethod
    def smart_glob(pattern, follow_symlinks=False):
        """Enhanced glob with additional options"""
        if follow_symlinks:
            # Custom implementation for following symlinks
            matches = []
            for root, dirs, files in os.walk('.', followlinks=True):
                for name in files + dirs:
                    path = os.path.join(root, name)
                    if fnmatch.fnmatch(path, pattern):
                        matches.append(path)
            return matches
        else:
            return glob.glob(pattern, recursive=True)

# Example usage
cross_glob = CrossPlatformGlob()

# Case insensitive search
case_insensitive_files = cross_glob.safe_glob('*.PY', case_sensitive=False)
print(f"Case insensitive Python files: {case_insensitive_files}")

# Normalized pattern
normalized_pattern = cross_glob.normalize_path_pattern('src/**/*.py')
print(f"Normalized pattern: {normalized_pattern}")

# Glob with exclusions
include_all = '**/*'
exclude_patterns = ['*.tmp', '*.log', '__pycache__/*', '.git/*']
filtered_files = cross_glob.glob_with_exclusions(include_all, exclude_patterns)
print(f"Filtered files (excluding temp/log): {len(filtered_files)}")

# Smart glob
smart_matches = cross_glob.smart_glob('**/*.py', follow_symlinks=True)
print(f"Smart glob matches: {len(smart_matches)}")
```

## Process and Signal Handling

### Signal Handling
```python
import signal
import os
import time
import sys

# Basic signal handling
def signal_handler(signum, frame):
    print(f"Received signal {signum}")
    if signum == signal.SIGINT:
        print("Interrupt signal received (Ctrl+C)")
        sys.exit(0)
    elif signum == signal.SIGTERM:
        print("Termination signal received")
        cleanup_and_exit()

def cleanup_and_exit():
    print("Performing cleanup...")
    # Cleanup code here
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # Termination request

# Unix/Linux/macOS specific signals
if hasattr(signal, 'SIGUSR1'):
    def user_signal_handler(signum, frame):
        print(f"User-defined signal {signum} received")
    
    signal.signal(signal.SIGUSR1, user_signal_handler)
    signal.signal(signal.SIGUSR2, user_signal_handler)

# Signal with timeout
class SignalTimeout:
    def __init__(self, seconds):
        self.seconds = seconds
    
    def __enter__(self):
        signal.signal(signal.SIGALRM, self._timeout_handler)
        signal.alarm(self.seconds)
        return self
    
    def __exit__(self, type, value, traceback):
        signal.alarm(0)  # Cancel alarm
    
    def _timeout_handler(self, signum, frame):
        raise TimeoutError(f"Operation timed out after {self.seconds} seconds")

# Example usage of signal timeout
try:
    with SignalTimeout(5):
        # Simulate long-running operation
        print("Starting long operation...")
        time.sleep(10)  # This will be interrupted
        print("Operation completed")
except TimeoutError as e:
    print(f"Timeout: {e}")

# Send signals to other processes
def send_signal_to_process(pid, sig=signal.SIGTERM):
    """Send signal to process"""
    try:
        os.kill(pid, sig)
        print(f"Signal {sig} sent to process {pid}")
        return True
    except ProcessLookupError:
        print(f"Process {pid} not found")
        return False
    except PermissionError:
        print(f"Permission denied to signal process {pid}")
        return False

# Example: find and signal processes by name
def find_processes_by_name(name):
    """Find processes by name (Unix/Linux/macOS)"""
    import subprocess
    try:
        result = subprocess.run(['pgrep', '-f', name], capture_output=True, text=True)
        if result.returncode == 0:
            pids = [int(pid) for pid in result.stdout.strip().split('\n') if pid]
            return pids
        return []
    except FileNotFoundError:
        # pgrep not available
        return []

# Signal all processes with name
def signal_processes_by_name(name, sig=signal.SIGTERM):
    """Signal all processes matching name"""
    pids = find_processes_by_name(name)
    for pid in pids:
        send_signal_to_process(pid, sig)
    return len(pids)

# Example usage
# matching_pids = find_processes_by_name('python')
# print(f"Found Python processes: {matching_pids}")
```

### Advanced Process Management
```python
import os
import signal
import subprocess
import threading
import time
import queue
from contextlib import contextmanager

class ProcessManager:
    def __init__(self):
        self.processes = {}
        self.monitoring = False
        self.monitor_thread = None
    
    def start_process(self, name, command, cwd=None, env=None):
        """Start a managed process"""
        try:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid  # Create new process group
            )
            
            self.processes[name] = {
                'process': process,
                'command': command,
                'start_time': time.time(),
                'restart_count': 0
            }
            
            print(f"Started process '{name}' with PID {process.pid}")
            return process.pid
        
        except Exception as e:
            print(f"Failed to start process '{name}': {e}")
            return None
    
    def stop_process(self, name, timeout=10):
        """Stop a managed process gracefully"""
        if name not in self.processes:
            return False
        
        process_info = self.processes[name]
        process = process_info['process']
        
        if process.poll() is not None:
            # Process already terminated
            del self.processes[name]
            return True
        
        try:
            # Send SIGTERM to process group
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            
            # Wait for graceful shutdown
            try:
                process.wait(timeout=timeout)
                print(f"Process '{name}' terminated gracefully")
            except subprocess.TimeoutExpired:
                # Force kill
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                process.wait()
                print(f"Process '{name}' force killed")
            
            del self.processes[name]
            return True
        
        except Exception as e:
            print(f"Error stopping process '{name}': {e}")
            return False
    
    def restart_process(self, name):
        """Restart a managed process"""
        if name not in self.processes:
            return False
        
        process_info = self.processes[name]
        command = process_info['command']
        
        # Stop the process
        self.stop_process(name)
        
        # Start it again
        pid = self.start_process(name, command)
        
        if pid and name in self.processes:
            self.processes[name]['restart_count'] += 1
            print(f"Restarted process '{name}' (restart #{self.processes[name]['restart_count']})")
            return True
        
        return False
    
    def get_process_status(self, name):
        """Get status of a managed process"""
        if name not in self.processes:
            return None
        
        process_info = self.processes[name]
        process = process_info['process']
        
        status = {
            'name': name,
            'pid': process.pid,
            'command': process_info['command'],
            'start_time': process_info['start_time'],
            'restart_count': process_info['restart_count'],
            'running': process.poll() is None
        }
        
        if not status['running']:
            status['exit_code'] = process.returncode
        
        return status
    
    def list_processes(self):
        """List all managed processes"""
        return [self.get_process_status(name) for name in self.processes.keys()]
    
    def start_monitoring(self, check_interval=5):
        """Start monitoring processes for crashes"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_processes, args=(check_interval,))
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop monitoring processes"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_processes(self, check_interval):
        """Monitor processes and restart if they crash"""
        while self.monitoring:
            crashed_processes = []
            
            for name in list(self.processes.keys()):
                status = self.get_process_status(name)
                if status and not status['running']:
                    print(f"Process '{name}' crashed with exit code {status.get('exit_code')}")
                    crashed_processes.append(name)
            
            # Restart crashed processes
            for name in crashed_processes:
                print(f"Restarting crashed process '{name}'")
                self.restart_process(name)
            
            time.sleep(check_interval)
    
    def shutdown_all(self):
        """Shutdown all managed processes"""
        print("Shutting down all processes...")
        for name in list(self.processes.keys()):
            self.stop_process(name)
        
        self.stop_monitoring()

# Context manager for process groups
@contextmanager
def process_group(processes, auto_restart=False):
    """Context manager for managing a group of processes"""
    manager = ProcessManager()
    
    try:
        # Start all processes
        for name, command in processes.items():
            manager.start_process(name, command)
        
        if auto_restart:
            manager.start_monitoring()
        
        yield manager
    
    finally:
        manager.shutdown_all()

# Example usage
if __name__ == "__main__":
    # Process group example
    processes = {
        'web_server': ['python3', '-m', 'http.server', '8000'],
        'background_worker': ['python3', '-c', 'import time; [time.sleep(1) for _ in range(60)]']
    }
    
    try:
        with process_group(processes, auto_restart=True) as manager:
            print("Process group started")
            
            # Monitor for a while
            time.sleep(10)
            
            # Check status
            for status in manager.list_processes():
                print(f"Process: {status}")
    
    except KeyboardInterrupt:
        print("Interrupted by user")
```

---

*This document covers comprehensive operating system and system programming functionality in Python including file/directory operations, process management, signal handling, path manipulation, temporary files, pattern matching, and advanced system administration tasks. For the most up-to-date information, refer to the official Python documentation.*