stage('Python Build & Test') {
        steps {
          sh '''
            # Use /tmp for cache (most containers have write access)
            export PIP_CACHE_DIR=/tmp/pip-cache
            mkdir -p $PIP_CACHE_DIR
            
            # Show cache directory contents before build
            echo "=== PIP Cache Directory Before Build ==="
            echo "Cache directory: $PIP_CACHE_DIR"
            ls -la $PIP_CACHE_DIR || echo "Cache directory is empty"
            du -sh $PIP_CACHE_DIR 2>/dev/null || echo "No cache data"
            
            # Create virtual environment
            echo "Creating virtual environment..."
            python -m venv .dcvirtualenv
            source ./.dcvirtualenv/bin/activate
            
            # Configure pip
            python -m pip install --upgrade pip
            pip config set global.cache-dir $PIP_CACHE_DIR
            
            # Install dependencies
            echo "Installing dependencies with pip cache..."
            python -m pip install --cache-dir=$PIP_CACHE_DIR --prefer-binary -r requirements.txt
            python -m pip install --cache-dir=$PIP_CACHE_DIR --prefer-binary -r requirements2.txt
            python -m pip install --cache-dir=$PIP_CACHE_DIR --prefer-binary -r requirements3.txt
            
            # Show final cache status
            echo "=== PIP Cache Directory After Build ==="
            ls -la $PIP_CACHE_DIR
            du -sh $PIP_CACHE_DIR
            
            # Verify environment
            echo "Python environment ready:"
            which python
            python --version
            pip list
            '''
        }
      }