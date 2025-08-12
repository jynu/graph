1. Update the CredentialManager class
Replace the existing CredentialManager class with this improved version:
pythonclass CredentialManager:
    """Handles secure credential management across platforms"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def _get_openssl_path(self) -> str:
        """Get OpenSSL executable path based on platform"""
        if os.name == 'nt':  # Windows
            # Common Windows OpenSSL locations
            possible_paths = [
                "C:/Program Files/OpenSSL-Win64/bin/openssl.exe",
                "C:/Program Files (x86)/OpenSSL-Win32/bin/openssl.exe",
                "C:/OpenSSL-Win64/bin/openssl.exe",
                "C:/OpenSSL-Win32/bin/openssl.exe",
                "openssl.exe",  # If in PATH
                "openssl"       # Fallback
            ]
        else:  # Linux/Unix
            possible_paths = [
                "/usr/local/bin/openssl",
                "/usr/bin/openssl",
                "/bin/openssl",
                "openssl"  # If in PATH
            ]
        
        # Check which OpenSSL is available
        for path in possible_paths:
            try:
                result = subprocess.run([path, "version"], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    self.logger.debug(f"Found OpenSSL at: {path}")
                    return path
            except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
                continue
        
        raise FileNotFoundError("OpenSSL executable not found on system")
    
    def decrypt_password(self) -> str:
        """Decrypt password using OpenSSL (cross-platform)"""
        encrypted_file = os.environ.get('ENCRYPTED_PASS_FILE')
        keyvalue_file = os.environ.get('KEY_VALUE_FILE')
        
        if not encrypted_file or not keyvalue_file:
            raise ValueError("Password decryption environment variables not set")
        
        if not os.path.exists(encrypted_file):
            raise FileNotFoundError(f"Encrypted password file not found: {encrypted_file}")
        
        if not os.path.exists(keyvalue_file):
            raise FileNotFoundError(f"Key value file not found: {keyvalue_file}")
        
        try:
            openssl_path = self._get_openssl_path()
            
            # OpenSSL command for decryption
            cmd = [
                openssl_path, "enc", "-aes-128-cbc", "-pbkdf2", 
                "-a", "-d", "-in", encrypted_file, "-pass", f"file:{keyvalue_file}"
            ]
            
            self.logger.debug(f"Executing decryption command on {os.name}")
            
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = proc.communicate()
            
            if proc.returncode != 0:
                error_msg = stderr.decode('utf-8').strip()
                raise RuntimeError(f"Password decryption failed: {error_msg}")
            
            decrypted_password = stdout.decode('utf-8').strip()
            
            if not decrypted_password:
                raise RuntimeError("Decrypted password is empty")
            
            self.logger.info(f"Password successfully decrypted on {os.name}")
            return decrypted_password
            
        except Exception as e:
            self.logger.error(f"Error decrypting password: {e}")
            raise
    
    def get_secure_password(self, provided_password: str = None) -> str:
        """Get password securely based on availability (cross-platform)"""
        
        # Priority 1: If password provided directly, use it (for backward compatibility)
        if provided_password:
            self.logger.info("Using provided password parameter")
            return provided_password
        
        # Priority 2: Try to decrypt from encrypted file (both Windows and Linux)
        try:
            decrypted_password = self.decrypt_password()
            if decrypted_password:
                self.logger.info(f"Using decrypted password from encrypted file ({os.name})")
                return decrypted_password
        except Exception as e:
            self.logger.warning(f"Could not decrypt password on {os.name}: {e}")
        
        # Priority 3: Try Windows Credential Manager (Windows only)
        if os.name == 'nt':
            try:
                windows_password = self._get_windows_credential()
                if windows_password:
                    self.logger.info("Using password from Windows Credential Manager")
                    return windows_password
            except Exception as e:
                self.logger.warning(f"Could not retrieve from Windows Credential Manager: {e}")
        
        # Priority 4: Fallback to interactive prompt
        import getpass
        self.logger.info("Prompting for password interactively")
        return getpass.getpass("Enter database password: ")
    
    def _get_windows_credential(self) -> str:
        """Get password from Windows Credential Manager (optional enhancement)"""
        try:
            import keyring
            service_name = "training_data_processor"
            username = os.environ.get('USERNAME', 'default_user')
            password = keyring.get_password(service_name, username)
            return password if password else ""
        except ImportError:
            self.logger.debug("keyring module not available for Windows Credential Manager")
            return ""
        except Exception as e:
            self.logger.debug(f"Windows Credential Manager access failed: {e}")
            return ""
2. Update environment validation for cross-platform
Replace the validate_environment() function:
pythondef validate_environment():
    """Validate environment setup for encrypted password (cross-platform)"""
    encrypted_file = os.environ.get('ENCRYPTED_PASS_FILE')
    keyvalue_file = os.environ.get('KEY_VALUE_FILE')
    
    print(f"Platform: {os.name} ({'Windows' if os.name == 'nt' else 'Linux/Unix'})")
    
    if encrypted_file and keyvalue_file:
        print("Encrypted password configuration detected:")
        print(f"  - Encrypted file: {encrypted_file}")
        print(f"  - Key file: {keyvalue_file}")
        
        # Check file existence
        if not os.path.exists(encrypted_file):
            print(f"  ⚠️  Warning: Encrypted password file not found: {encrypted_file}")
        else:
            print(f"  ✓ Encrypted password file found")
        
        if not os.path.exists(keyvalue_file):
            print(f"  ⚠️  Warning: Key value file not found: {keyvalue_file}")
        else:
            print(f"  ✓ Key file found")
        
        # Check OpenSSL availability
        try:
            credential_manager = CredentialManager()
            openssl_path = credential_manager._get_openssl_path()
            print(f"  ✓ OpenSSL found at: {openssl_path}")
        except FileNotFoundError:
            print(f"  ⚠️  Warning: OpenSSL not found on system")
    else:
        print("No encrypted password configuration detected")
        if not encrypted_file:
            print("  - ENCRYPTED_PASS_FILE environment variable not set")
        if not keyvalue_file:
            print("  - KEY_VALUE_FILE environment variable not set")
        print("  - Will fall back to interactive password prompt")
3. Add OpenSSL installation guidance in configuration comments
Add this to the configuration section:
python# =============================================================================
# CREDENTIAL MANAGEMENT SETUP
# =============================================================================

# For encrypted password support, set these environment variables:
# export ENCRYPTED_PASS_FILE="/path/to/encrypted_password.enc"
# export KEY_VALUE_FILE="/path/to/keyfile.key"
#
# OpenSSL Requirements:
# Windows: Install OpenSSL from https://slproweb.com/products/Win32OpenSSL.html
#          or use chocolatey: choco install openssl
# Linux:   Usually pre-installed, or: sudo apt-get install openssl
#
# To create encrypted password file:
# echo "your_password" | openssl enc -aes-128-cbc -pbkdf2 -a -out encrypted_password.enc -pass file:keyfile.key
🚀 Usage Examples:
Windows with Encrypted Password:
batchREM Set environment variables
set ENCRYPTED_PASS_FILE=C:\secure\encrypted_password.enc
set KEY_VALUE_FILE=C:\secure\keyfile.key

REM Run without password argument
python 1_get_frequent_columns.py --table "SCHEMA.TABLE" --user username
Linux with Encrypted Password:
bash# Set environment variables
export ENCRYPTED_PASS_FILE="/secure/encrypted_password.enc"
export KEY_VALUE_FILE="/secure/keyfile.key"

# Run without password argument
python 1_get_frequent_columns.py --table "SCHEMA.TABLE" --user username
Creating Encrypted Password Files (Cross-Platform):
bash# Create a key file (do this once)
openssl rand -base64 32 > keyfile.key

# Encrypt your password
echo "your_actual_password" | openssl enc -aes-128-cbc -pbkdf2 -a -out encrypted_password.enc -pass file:keyfile.key
🔐 Enhanced Security Features: