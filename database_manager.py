# database_manager.py
"""
Multi-Database Configuration and Connection Manager
==================================================
A comprehensive database abstraction layer supporting multiple database types
including Impala, Oracle, and Sybase for the training data processing pipeline.

This module provides:
1. Database-specific configuration management
2. Unified connection interface
3. Database-specific SQL query adaptation
4. Connection pooling and error handling
5. Cross-platform credential management

Supported Databases:
- Impala (via JDBC/HVApi)
- Oracle (via JDBC/native drivers)  
- Sybase (planned)
"""

import os
import sys
import json
import logging
import subprocess
import tempfile
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import configparser

# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class DatabaseType(Enum):
    """Supported database types"""
    IMPALA = "impala"
    ORACLE = "oracle"
    SYBASE = "sybase"

class Environment(Enum):
    """Environment types"""
    DEV = "dev"
    UAT = "uat"
    PROD = "prod"

# =============================================================================
# CONFIGURATION CLASSES
# =============================================================================

@dataclass
class DatabaseConnectionConfig:
    """Base database connection configuration"""
    db_type: DatabaseType
    environment: Environment
    host: str
    port: Optional[int] = None
    database: str = ""
    user_id: str = ""
    password: str = ""
    connection_properties: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.connection_properties is None:
            self.connection_properties = {}

@dataclass
class SystemPaths:
    """System-specific paths for database connectivity"""
    java_path: str
    jar_path: str
    driver_path: Optional[str] = None
    
    def validate(self) -> bool:
        """Validate that required paths exist"""
        # For Oracle, we don't need to validate paths since we use oracledb library
        if not self.java_path and not self.jar_path:
            return True  # Oracle case - no paths needed
        
        paths_to_check = [self.java_path, self.jar_path]
        if self.driver_path:
            paths_to_check.append(self.driver_path)
        
        for path in paths_to_check:
            if path and not os.path.exists(path):
                return False
        return True

# =============================================================================
# DATABASE CONFIGURATION FACTORY
# =============================================================================

class DatabaseConfigFactory:
    """Factory for creating database-specific configurations"""
    
    # Database connection templates
    CONNECTION_CONFIGS = {
        DatabaseType.IMPALA: {
            Environment.PROD: {
                "host": "--application.server.host=ws://olympus-high-volume-api-icg-isg-olympus-high-volume-api-167969.apps.namicgrut37p.ecs.dyn.nsroot.net",
                "port": None,
                "database": "",
                "connection_properties": {
                    "format": "TXT",
                    "timeout": 300
                }
            },
            Environment.DEV: {
                "host": "--application.server.host=ws://olympus-high-volume-api-icg-isg-olympus-high-volume-api-167969.apps.namicggtd35d.ecs.dyn.nsroot.net",
                "port": None,
                "database": "",
                "connection_properties": {
                    "format": "TXT",
                    "timeout": 300
                }
            }
        },
        DatabaseType.ORACLE: {
            Environment.DEV: {
                "host": "OLY2DEV.oraas.dyn.nsroot.net",
                "port": 8889,
                "database": "haOLY2DEV",
                "connection_properties": {
                    "driver": "oracle.jdbc.driver.OracleDriver",
                    "connection_string_template": "jdbc:oracle:thin:@{host}:{port}/{database}"
                }
            },
            Environment.UAT: {
                "host": "OLY2UAT.oraas.dyn.nsroot.net", 
                "port": 8889,
                "database": "haOLY2UAT",
                "connection_properties": {
                    "driver": "oracle.jdbc.driver.OracleDriver",
                    "connection_string_template": "jdbc:oracle:thin:@{host}:{port}/{database}"
                }
            }
        },
        DatabaseType.SYBASE: {
            # Placeholder for future Sybase configuration
            Environment.DEV: {
                "host": "sybase-dev.example.com",
                "port": 5000,
                "database": "training_data",
                "connection_properties": {
                    "driver": "com.sybase.jdbc4.jdbc.SybDriver",
                    "connection_string_template": "jdbc:sybase:Tds:{host}:{port}/{database}"
                }
            }
        }
    }
    
    # System paths by platform and database type
    SYSTEM_PATHS = {
        DatabaseType.IMPALA: {
            "windows": {
                "java_path": "C:/work/training_data/jdk-17.0.7/jdk-17.0.7/bin/java.exe",
                "jar_path": "C:/work/training_data/mktdata-report-hvapi-commandline-client-1.0.30.jar"
            },
            "linux": {
                "java_path": "/opt/jdk/17.0_9l64/bin/java",
                "jar_path": "/home/bj33244/mktdata-report-hvapi-commandline-client-1.0.27-SNAPSHOT.jar"
            }
        },
        DatabaseType.ORACLE: {
            # Oracle uses oracledb library, no system paths needed
            "windows": {
                "java_path": "",  # Not used for Oracle
                "jar_path": ""    # Not used for Oracle
            },
            "linux": {
                "java_path": "",  # Not used for Oracle
                "jar_path": ""    # Not used for Oracle
            }
        }
    }
    
    @classmethod
    def create_config(cls, db_type: DatabaseType, environment: Environment, 
                     user_id: str, password: str) -> DatabaseConnectionConfig:
        """Create database configuration for specified type and environment"""
        
        if db_type not in cls.CONNECTION_CONFIGS:
            raise ValueError(f"Unsupported database type: {db_type}")
        
        if environment not in cls.CONNECTION_CONFIGS[db_type]:
            raise ValueError(f"Unsupported environment {environment} for database {db_type}")
        
        config_template = cls.CONNECTION_CONFIGS[db_type][environment]
        
        return DatabaseConnectionConfig(
            db_type=db_type,
            environment=environment,
            host=config_template["host"],
            port=config_template.get("port"),
            database=config_template.get("database", ""),
            user_id=user_id,
            password=password,
            connection_properties=config_template.get("connection_properties", {})
        )
    
    @classmethod
    def get_system_paths(cls, db_type: DatabaseType) -> SystemPaths:
        """Get system paths for database type and current platform"""
        platform = "windows" if os.name == 'nt' else "linux"
        
        if db_type not in cls.SYSTEM_PATHS:
            raise ValueError(f"No system paths configured for database type: {db_type}")
        
        if platform not in cls.SYSTEM_PATHS[db_type]:
            raise ValueError(f"No system paths configured for {db_type} on {platform}")
        
        paths_config = cls.SYSTEM_PATHS[db_type][platform]
        
        return SystemPaths(
            java_path=paths_config["java_path"],
            jar_path=paths_config["jar_path"],
            driver_path=paths_config.get("driver_path")
        )

# =============================================================================
# SQL DIALECT ADAPTERS
# =============================================================================

class SQLDialectAdapter(ABC):
    """Abstract base class for database-specific SQL adaptations"""
    
    @abstractmethod
    def format_table_describe(self, table_name: str) -> str:
        """Format table description query"""
        pass
    
    @abstractmethod
    def format_distinct_count(self, table_name: str, column_name: str, 
                            where_conditions: str) -> str:
        """Format distinct count query"""
        pass
    
    @abstractmethod
    def format_distinct_values(self, table_name: str, column_name: str, 
                             where_conditions: str, limit: int) -> str:
        """Format distinct values query"""
        pass
    
    @abstractmethod
    def format_date_condition(self, column_name: str, column_type: str, 
                            start_date: datetime, end_date: datetime) -> str:
        """Format date condition based on column type"""
        pass
    
    @abstractmethod
    def map_column_type(self, db_column_type: str) -> str:
        """Map database-specific column type to standardized type"""
        pass

class ImpalaDialectAdapter(SQLDialectAdapter):
    """SQL dialect adapter for Impala"""
    
    def format_table_describe(self, table_name: str) -> str:
        return f"DESCRIBE {table_name}"
    
    def format_distinct_count(self, table_name: str, column_name: str, 
                            where_conditions: str) -> str:
        return f"SELECT COUNT(DISTINCT {column_name}) FROM {table_name} WHERE {where_conditions}"
    
    def format_distinct_values(self, table_name: str, column_name: str, 
                             where_conditions: str, limit: int) -> str:
        return f"SELECT DISTINCT {column_name} FROM {table_name} WHERE {where_conditions} LIMIT {limit}"
    
    def format_date_condition(self, column_name: str, column_type: str, 
                            start_date: datetime, end_date: datetime) -> str:
        if "string" in column_type.lower():
            start_str = f"'{start_date.strftime('%Y%m%d')}'"
            end_str = f"'{end_date.strftime('%Y%m%d')}'"
        elif "timestamp" in column_type.lower():
            start_str = f"CAST('{start_date.strftime('%Y-%m-%d %H:%M:%S')}' AS TIMESTAMP)"
            end_str = f"CAST('{end_date.strftime('%Y-%m-%d %H:%M:%S')}' AS TIMESTAMP)"
        else:
            start_str = start_date.strftime('%Y%m%d')
            end_str = end_date.strftime('%Y%m%d')
        
        return f"{column_name} > {start_str} AND {column_name} < {end_str}"
    
    def map_column_type(self, db_column_type: str) -> str:
        column_type_lower = db_column_type.lower()
        
        if any(t in column_type_lower for t in ['varchar', 'string', 'char']):
            return "string"
        elif any(t in column_type_lower for t in ['bigint', 'int']):
            return "integer"
        elif any(t in column_type_lower for t in ['decimal', 'double']):
            return "float"
        elif 'timestamp' in column_type_lower:
            return "timestamp"
        elif column_type_lower.startswith('map<'):
            return "nestedtype"
        else:
            return "string"

class OracleDialectAdapter(SQLDialectAdapter):
    """SQL dialect adapter for Oracle"""
    
    def format_table_describe(self, table_name: str) -> str:
        # Oracle uses ALL_TAB_COLUMNS or DESC
        schema_table = table_name.split('.')
        if len(schema_table) == 2:
            schema, table = schema_table
            return f"""SELECT COLUMN_NAME, DATA_TYPE 
                      FROM ALL_TAB_COLUMNS 
                      WHERE OWNER = '{schema.upper()}' 
                      AND TABLE_NAME = '{table.upper()}'
                      ORDER BY COLUMN_ID"""
        else:
            return f"DESC {table_name}"
    
    def format_distinct_count(self, table_name: str, column_name: str, 
                            where_conditions: str) -> str:
        return f"SELECT COUNT(DISTINCT {column_name}) FROM {table_name} WHERE {where_conditions}"
    
    def format_distinct_values(self, table_name: str, column_name: str, 
                             where_conditions: str, limit: int) -> str:
        # Oracle uses ROWNUM instead of LIMIT
        return f"""SELECT DISTINCT {column_name} 
                  FROM {table_name} 
                  WHERE {where_conditions} 
                  AND ROWNUM <= {limit}"""
    
    def format_date_condition(self, column_name: str, column_type: str, 
                            start_date: datetime, end_date: datetime) -> str:
        if "varchar" in column_type.lower() or "char" in column_type.lower():
            start_str = f"'{start_date.strftime('%Y%m%d')}'"
            end_str = f"'{end_date.strftime('%Y%m%d')}'"
        elif "date" in column_type.lower() or "timestamp" in column_type.lower():
            start_str = f"TO_DATE('{start_date.strftime('%Y-%m-%d')}', 'YYYY-MM-DD')"
            end_str = f"TO_DATE('{end_date.strftime('%Y-%m-%d')}', 'YYYY-MM-DD')"
        else:
            start_str = start_date.strftime('%Y%m%d')
            end_str = end_date.strftime('%Y%m%d')
        
        return f"{column_name} > {start_str} AND {column_name} < {end_str}"
    
    def map_column_type(self, db_column_type: str) -> str:
        column_type_lower = db_column_type.lower()
        
        if any(t in column_type_lower for t in ['varchar', 'char', 'clob']):
            return "string"
        elif any(t in column_type_lower for t in ['number', 'integer']):
            return "integer"
        elif any(t in column_type_lower for t in ['float', 'decimal']):
            return "float"
        elif any(t in column_type_lower for t in ['date', 'timestamp']):
            return "timestamp"
        else:
            return "string"

class SybaseDialectAdapter(SQLDialectAdapter):
    """SQL dialect adapter for Sybase (placeholder implementation)"""
    
    def format_table_describe(self, table_name: str) -> str:
        return f"sp_columns {table_name}"
    
    def format_distinct_count(self, table_name: str, column_name: str, 
                            where_conditions: str) -> str:
        return f"SELECT COUNT(DISTINCT {column_name}) FROM {table_name} WHERE {where_conditions}"
    
    def format_distinct_values(self, table_name: str, column_name: str, 
                             where_conditions: str, limit: int) -> str:
        # Sybase uses TOP instead of LIMIT
        return f"SELECT DISTINCT TOP {limit} {column_name} FROM {table_name} WHERE {where_conditions}"
    
    def format_date_condition(self, column_name: str, column_type: str, 
                            start_date: datetime, end_date: datetime) -> str:
        if "varchar" in column_type.lower() or "char" in column_type.lower():
            start_str = f"'{start_date.strftime('%Y%m%d')}'"
            end_str = f"'{end_date.strftime('%Y%m%d')}'"
        elif "datetime" in column_type.lower():
            start_str = f"'{start_date.strftime('%Y-%m-%d %H:%M:%S')}'"
            end_str = f"'{end_date.strftime('%Y-%m-%d %H:%M:%S')}'"
        else:
            start_str = start_date.strftime('%Y%m%d')
            end_str = end_date.strftime('%Y%m%d')
        
        return f"{column_name} > {start_str} AND {column_name} < {end_str}"
    
    def map_column_type(self, db_column_type: str) -> str:
        column_type_lower = db_column_type.lower()
        
        if any(t in column_type_lower for t in ['varchar', 'char', 'text']):
            return "string"
        elif any(t in column_type_lower for t in ['int', 'bigint', 'smallint']):
            return "integer"
        elif any(t in column_type_lower for t in ['float', 'decimal', 'numeric']):
            return "float"
        elif any(t in column_type_lower for t in ['datetime', 'timestamp']):
            return "timestamp"
        else:
            return "string"

# =============================================================================
# DATABASE EXECUTOR CLASSES
# =============================================================================

class DatabaseExecutor(ABC):
    """Abstract base class for database-specific query execution"""
    
    def __init__(self, config: DatabaseConnectionConfig, system_paths: SystemPaths):
        self.config = config
        self.system_paths = system_paths
        self.logger = logging.getLogger(__name__)
        self.dialect_adapter = self._get_dialect_adapter()
    
    @abstractmethod
    def _get_dialect_adapter(self) -> SQLDialectAdapter:
        """Get database-specific SQL dialect adapter"""
        pass
    
    @abstractmethod
    def execute_query(self, query: str, output_file: str, format_type: str = "TXT") -> bool:
        """Execute query and save results to file"""
        pass
    
    def get_table_schema(self, table_name: str) -> Dict[str, str]:
        """Get table schema information"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.schema', delete=False) as temp_file:
            schema_file = temp_file.name
        
        try:
            query = self.dialect_adapter.format_table_describe(table_name)
            if self.execute_query(query, schema_file):
                return self._parse_schema_file(schema_file)
            return {}
        finally:
            if os.path.exists(schema_file):
                os.unlink(schema_file)
    
    def _parse_schema_file(self, schema_file: str) -> Dict[str, str]:
        """Parse schema file and return column type mappings"""
        schema_dict = {}
        
        try:
            with open(schema_file, 'r') as f:
                lines = f.readlines()
            
            for i, line in enumerate(lines):
                if i == 0:  # Skip header
                    continue
                
                parts = line.strip().split(',', 1)
                if len(parts) >= 2:
                    column_name = parts[0].strip().lower()
                    column_type = parts[1].strip().rstrip(',')
                    mapped_type = self.dialect_adapter.map_column_type(column_type)
                    schema_dict[column_name] = f"{column_type}|{mapped_type}"
            
            return schema_dict
            
        except Exception as e:
            self.logger.error(f"Error parsing schema file: {e}")
            return {}

class ImpalaExecutor(DatabaseExecutor):
    """Impala-specific query executor using JDBC/HVApi"""
    
    def _get_dialect_adapter(self) -> SQLDialectAdapter:
        return ImpalaDialectAdapter()
    
    def execute_query(self, query: str, output_file: str, format_type: str = "TXT") -> bool:
        """Execute Impala query using existing JDBC client"""
        try:
            command = [
                self.system_paths.java_path,
                "--add-opens=java.base/java.nio=ALL-UNNAMED",
                "-jar", self.system_paths.jar_path,
                self.config.host,
                f"--query={query}",
                f"--user={self.config.user_id}",
                f"--pass={self.config.password}",
                f"--env={self.config.environment.value}",
                f"--format={format_type}",
                f"--destination={output_file}"
            ]
            
            self.logger.info(f"Executing Impala query: {query}")
            
            # Validate paths
            if not self.system_paths.validate():
                self.logger.error("System paths validation failed")
                return False
            
            result = subprocess.run(command, capture_output=True, text=True, 
                                  timeout=self.config.connection_properties.get('timeout', 300))
            
            if result.returncode != 0:
                self.logger.error(f"Impala query failed: {result.stderr}")
                return False
            
            return os.path.exists(output_file) and os.path.getsize(output_file) > 0
            
        except Exception as e:
            self.logger.error(f"Error executing Impala query: {e}")
            return False

class OracleExecutor(DatabaseExecutor):
    """Oracle-specific query executor using oracledb library"""
    
    def __init__(self, config: DatabaseConnectionConfig, system_paths: SystemPaths):
        super().__init__(config, system_paths)
        self.connection = None
        self.cursor = None
        
        # Import oracledb library
        try:
            import oracledb
            self.oracledb = oracledb
            self.logger.info("Oracle client library loaded successfully")
        except ImportError:
            self.logger.error("oracledb library not found. Install with: pip install oracledb")
            raise ImportError("oracledb library is required for Oracle connections")
    
    def _get_dialect_adapter(self) -> SQLDialectAdapter:
        return OracleDialectAdapter()
    
    def _connect(self) -> bool:
        """Establish Oracle database connection"""
        try:
            if self.connection and not self.connection.ping():
                self.connection = None
            
            if not self.connection:
                # Build DSN from configuration
                dsn = f"{self.config.host}:{self.config.port}/{self.config.database}"
                
                self.logger.info(f"Connecting to Oracle: {dsn}")
                self.connection = self.oracledb.connect(
                    user=self.config.user_id,
                    password=self.config.password,
                    dsn=dsn
                )
                self.cursor = self.connection.cursor()
                self.logger.info("Oracle connection established successfully")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Oracle connection failed: {e}")
            return False
    
    def _disconnect(self):
        """Close Oracle connection"""
        try:
            if self.cursor:
                self.cursor.close()
                self.cursor = None
            if self.connection:
                self.connection.close()
                self.connection = None
        except Exception as e:
            self.logger.warning(f"Error closing Oracle connection: {e}")
    
    def execute_query(self, query: str, output_file: str, format_type: str = "TXT") -> bool:
        """Execute Oracle query using oracledb library"""
        try:
            self.logger.info(f"Executing Oracle query: {query}")
            
            # Establish connection
            if not self._connect():
                return False
            
            # Execute query
            self.cursor.execute(query)
            results = self.cursor.fetchall()
            
            # Get column names
            column_names = [desc[0] for desc in self.cursor.description] if self.cursor.description else []
            
            # Write results to file
            with open(output_file, 'w', encoding='utf-8') as f:
                # Write header
                if column_names:
                    f.write(','.join(column_names) + '\n')
                
                # Write data rows
                for row in results:
                    # Convert row values to strings and handle None values
                    row_values = []
                    for value in row:
                        if value is None:
                            row_values.append('')
                        elif isinstance(value, (int, float)):
                            row_values.append(str(value))
                        elif hasattr(value, 'strftime'):  # Date/datetime objects
                            row_values.append(value.strftime('%Y-%m-%d %H:%M:%S'))
                        else:
                            row_values.append(str(value))
                    
                    f.write(','.join(row_values) + '\n')
            
            self.logger.info(f"Query executed successfully, {len(results)} rows written to {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error executing Oracle query: {e}")
            return False
        finally:
            # Keep connection open for reuse but close cursor
            if self.cursor:
                try:
                    self.cursor.close()
                    self.cursor = None
                except:
                    pass
    
    def get_table_schema(self, table_name: str) -> Dict[str, str]:
        """Get Oracle table schema information"""
        try:
            if not self._connect():
                return {}
            
            # Use Oracle-specific schema query
            schema_query = self.dialect_adapter.format_table_describe(table_name)
            
            self.cursor.execute(schema_query)
            results = self.cursor.fetchall()
            
            schema_dict = {}
            for row in results:
                if len(row) >= 2:
                    column_name = str(row[0]).lower()
                    column_type = str(row[1])
                    mapped_type = self.dialect_adapter.map_column_type(column_type)
                    schema_dict[column_name] = f"{column_type}|{mapped_type}"
            
            self.logger.info(f"Retrieved schema for {table_name}: {len(schema_dict)} columns")
            return schema_dict
            
        except Exception as e:
            self.logger.error(f"Error getting Oracle table schema: {e}")
            return {}
        finally:
            if self.cursor:
                try:
                    self.cursor.close()
                    self.cursor = None
                except:
                    pass
    
    def __del__(self):
        """Cleanup connection on object destruction"""
        self._disconnect()

class SybaseExecutor(DatabaseExecutor):
    """Sybase-specific query executor (placeholder implementation)"""
    
    def _get_dialect_adapter(self) -> SQLDialectAdapter:
        return SybaseDialectAdapter()
    
    def execute_query(self, query: str, output_file: str, format_type: str = "TXT") -> bool:
        """Execute Sybase query (placeholder implementation)"""
        self.logger.warning("Sybase executor not yet implemented")
        return False

# =============================================================================
# MAIN DATABASE MANAGER
# =============================================================================

class MultiDatabaseManager:
    """
    Main database manager that provides unified interface for multiple database types
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._executors = {}
        
        # Register available executors
        self._executor_classes = {
            DatabaseType.IMPALA: ImpalaExecutor,
            DatabaseType.ORACLE: OracleExecutor,
            DatabaseType.SYBASE: SybaseExecutor
        }
    
    def get_database_client(self, db_type: DatabaseType, environment: Environment, 
                          user_id: str, password: str) -> DatabaseExecutor:
        """
        Get database client for specified database type and environment
        """
        try:
            # Create unique key for caching
            cache_key = f"{db_type.value}_{environment.value}_{user_id}"
            
            if cache_key not in self._executors:
                # Create database configuration
                config = DatabaseConfigFactory.create_config(db_type, environment, user_id, password)
                system_paths = DatabaseConfigFactory.get_system_paths(db_type)
                
                # Validate system paths
                if not system_paths.validate():
                    raise RuntimeError(f"System paths validation failed for {db_type.value}")
                
                # Create executor
                executor_class = self._executor_classes.get(db_type)
                if not executor_class:
                    raise ValueError(f"No executor available for database type: {db_type}")
                
                executor = executor_class(config, system_paths)
                self._executors[cache_key] = executor
                
                self.logger.info(f"Created database client for {db_type.value} {environment.value}")
            
            return self._executors[cache_key]
            
        except Exception as e:
            self.logger.error(f"Failed to create database client: {e}")
            raise
    
    def get_supported_databases(self) -> List[DatabaseType]:
        """Get list of supported database types"""
        return list(self._executor_classes.keys())
    
    def get_supported_environments(self, db_type: DatabaseType) -> List[Environment]:
        """Get list of supported environments for a database type"""
        if db_type in DatabaseConfigFactory.CONNECTION_CONFIGS:
            return list(DatabaseConfigFactory.CONNECTION_CONFIGS[db_type].keys())
        return []
    
    def test_connection(self, db_type: DatabaseType, environment: Environment, 
                       user_id: str, password: str) -> bool:
        """Test database connection"""
        try:
            client = self.get_database_client(db_type, environment, user_id, password)
            
            # Test with simple query
            with tempfile.NamedTemporaryFile(mode='w', suffix='.test', delete=False) as temp_file:
                temp_output = temp_file.name
            
            try:
                test_query = "SELECT 1 as test"
                result = client.execute_query(test_query, temp_output)
                
                if result and os.path.exists(temp_output) and os.path.getsize(temp_output) > 0:
                    self.logger.info(f"Connection test successful for {db_type.value} {environment.value}")
                    return True
                else:
                    self.logger.error(f"Connection test failed for {db_type.value} {environment.value}")
                    return False
                    
            finally:
                if os.path.exists(temp_output):
                    os.unlink(temp_output)
                    
        except Exception as e:
            self.logger.error(f"Connection test error for {db_type.value} {environment.value}: {e}")
            return False
    
    def validate_database_config(self, db_type: DatabaseType, environment: Environment) -> bool:
        """Validate that database configuration exists"""
        try:
            if db_type not in DatabaseConfigFactory.CONNECTION_CONFIGS:
                return False
            
            if environment not in DatabaseConfigFactory.CONNECTION_CONFIGS[db_type]:
                return False
            
            # Check system paths
            system_paths = DatabaseConfigFactory.get_system_paths(db_type)
            return system_paths.validate()
            
        except Exception:
            return False

# =============================================================================
# CREDENTIAL MANAGER (ENHANCED)
# =============================================================================

class EnhancedCredentialManager:
    """Enhanced credential manager with multi-database support"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def get_credentials(self, db_type: DatabaseType, environment: Environment, 
                       provided_user: str = None, provided_password: str = None) -> tuple:
        """Get credentials for specific database and environment"""
        
        # Priority 1: Use provided credentials
        if provided_user and provided_password:
            self.logger.info(f"Using provided credentials for {db_type.value} {environment.value}")
            return provided_user, provided_password
        
        # Priority 2: Try environment-specific encrypted files
        try:
            env_user, env_password = self._get_environment_credentials(db_type, environment)
            if env_user and env_password:
                self.logger.info(f"Using environment-specific encrypted credentials for {db_type.value} {environment.value}")
                return env_user, env_password
        except Exception as e:
            self.logger.warning(f"Could not get environment credentials: {e}")
        
        # Priority 3: Try generic encrypted files
        try:
            from get_frequent_columns import CredentialManager
            generic_cred_manager = CredentialManager()
            generic_password = generic_cred_manager.get_secure_password(provided_password)
            if provided_user and generic_password:
                self.logger.info(f"Using generic encrypted credentials for {db_type.value} {environment.value}")
                return provided_user, generic_password
        except Exception as e:
            self.logger.warning(f"Could not get generic encrypted credentials: {e}")
        
        # Priority 4: Interactive prompt
        import getpass
        self.logger.info(f"Prompting for credentials for {db_type.value} {environment.value}")
        user = provided_user or input(f"Enter username for {db_type.value} {environment.value}: ")
        password = getpass.getpass(f"Enter password for {db_type.value} {environment.value}: ")
        
        return user, password
    
    def _get_environment_credentials(self, db_type: DatabaseType, environment: Environment) -> tuple:
        """Get environment-specific credentials from encrypted files"""
        # Look for environment-specific credential files
        env_prefix = f"{db_type.value.upper()}_{environment.value.upper()}"
        
        encrypted_file = os.environ.get(f'{env_prefix}_ENCRYPTED_PASS_FILE')
        key_file = os.environ.get(f'{env_prefix}_KEY_VALUE_FILE')
        user_env = os.environ.get(f'{env_prefix}_USERNAME')
        
        if encrypted_file and key_file and user_env:
            # Use the existing credential manager with environment-specific files
            original_encrypted = os.environ.get('ENCRYPTED_PASS_FILE')
            original_key = os.environ.get('KEY_VALUE_FILE')
            
            try:
                # Temporarily set environment variables
                os.environ['ENCRYPTED_PASS_FILE'] = encrypted_file
                os.environ['KEY_VALUE_FILE'] = key_file
                
                from get_frequent_columns import CredentialManager
                cred_manager = CredentialManager()
                password = cred_manager.decrypt_password()
                
                return user_env, password
                
            finally:
                # Restore original environment variables
                if original_encrypted:
                    os.environ['ENCRYPTED_PASS_FILE'] = original_encrypted
                elif 'ENCRYPTED_PASS_FILE' in os.environ:
                    del os.environ['ENCRYPTED_PASS_FILE']
                
                if original_key:
                    os.environ['KEY_VALUE_FILE'] = original_key
                elif 'KEY_VALUE_FILE' in os.environ:
                    del os.environ['KEY_VALUE_FILE']
        
        return None, None

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_database_manager() -> MultiDatabaseManager:
    """Factory function to create database manager instance"""
    return MultiDatabaseManager()

def get_database_client_from_string(database_spec: str, user_id: str, password: str) -> DatabaseExecutor:
    """
    Create database client from string specification
    Format: "database_type:environment" (e.g., "impala:prod", "oracle:dev")
    """
    try:
        parts = database_spec.lower().split(':')
        if len(parts) != 2:
            raise ValueError("Database spec must be in format 'type:environment'")
        
        db_type_str, env_str = parts
        
        # Parse database type
        db_type = None
        for dt in DatabaseType:
            if dt.value == db_type_str:
                db_type = dt
                break
        
        if not db_type:
            raise ValueError(f"Unsupported database type: {db_type_str}")
        
        # Parse environment
        environment = None
        for env in Environment:
            if env.value == env_str:
                environment = env
                break
        
        if not environment:
            raise ValueError(f"Unsupported environment: {env_str}")
        
        # Create manager and get client
        manager = create_database_manager()
        return manager.get_database_client(db_type, environment, user_id, password)
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to create database client from spec '{database_spec}': {e}")
        raise

def validate_database_specification(database_spec: str) -> bool:
    """Validate database specification string"""
    try:
        parts = database_spec.lower().split(':')
        if len(parts) != 2:
            return False
        
        db_type_str, env_str = parts
        
        # Check if database type is supported
        db_types = [dt.value for dt in DatabaseType]
        if db_type_str not in db_types:
            return False
        
        # Check if environment is supported
        environments = [env.value for env in Environment]
        if env_str not in environments:
            return False
        
        return True
        
    except Exception:
        return False

def list_available_databases() -> Dict[str, List[str]]:
    """List all available database configurations"""
    result = {}
    
    for db_type in DatabaseType:
        environments = []
        if db_type in DatabaseConfigFactory.CONNECTION_CONFIGS:
            environments = [env.value for env in DatabaseConfigFactory.CONNECTION_CONFIGS[db_type].keys()]
        result[db_type.value] = environments
    
    return result

# =============================================================================
# CONFIGURATION FILE SUPPORT
# =============================================================================

class DatabaseConfigLoader:
    """Load database configurations from external config files"""
    
    def __init__(self, config_file_path: str = None):
        self.config_file_path = config_file_path or "database_config.ini"
        self.logger = logging.getLogger(__name__)
    
    def load_custom_config(self) -> bool:
        """Load custom database configurations from file"""
        try:
            if not os.path.exists(self.config_file_path):
                self.logger.info(f"No custom config file found at {self.config_file_path}")
                return False
            
            config = configparser.ConfigParser()
            config.read(self.config_file_path)
            
            # Update CONNECTION_CONFIGS with custom configurations
            for section_name in config.sections():
                try:
                    self._parse_database_section(config, section_name)
                except Exception as e:
                    self.logger.warning(f"Failed to parse config section {section_name}: {e}")
            
            self.logger.info(f"Loaded custom database configurations from {self.config_file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading custom config: {e}")
            return False
    
    def _parse_database_section(self, config: configparser.ConfigParser, section_name: str):
        """Parse individual database configuration section"""
        # Expected format: [database_type.environment]
        parts = section_name.split('.')
        if len(parts) != 2:
            raise ValueError(f"Invalid section name format: {section_name}")
        
        db_type_str, env_str = parts
        
        # Convert to enums
        db_type = DatabaseType(db_type_str.lower())
        environment = Environment(env_str.lower())
        
        # Extract configuration
        section_config = {
            "host": config.get(section_name, "host"),
            "port": config.getint(section_name, "port") if config.has_option(section_name, "port") else None,
            "database": config.get(section_name, "database", fallback=""),
            "connection_properties": {}
        }
        
        # Extract connection properties
        for option in config.options(section_name):
            if option.startswith("prop_"):
                prop_name = option[5:]  # Remove "prop_" prefix
                section_config["connection_properties"][prop_name] = config.get(section_name, option)
        
        # Update global configuration
        if db_type not in DatabaseConfigFactory.CONNECTION_CONFIGS:
            DatabaseConfigFactory.CONNECTION_CONFIGS[db_type] = {}
        
        DatabaseConfigFactory.CONNECTION_CONFIGS[db_type][environment] = section_config

# =============================================================================
# MAIN CLI INTERFACE (for testing)
# =============================================================================

def main():
    """Main CLI interface for testing database connections"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Multi-Database Connection Manager')
    parser.add_argument('--database', '-d', type=str, required=True,
                       help='Database specification (format: type:environment, e.g., impala:prod)')
    parser.add_argument('--user', '-u', type=str, required=True,
                       help='Database username')
    parser.add_argument('--password', '-p', type=str,
                       help='Database password (optional, will prompt if not provided)')
    parser.add_argument('--test', '-t', action='store_true',
                       help='Test database connection')
    parser.add_argument('--list', '-l', action='store_true',
                       help='List available database configurations')
    parser.add_argument('--query', '-q', type=str,
                       help='Execute test query')
    parser.add_argument('--config', '-c', type=str,
                       help='Custom configuration file path')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        # Load custom configuration if provided
        if args.config:
            config_loader = DatabaseConfigLoader(args.config)
            config_loader.load_custom_config()
        
        # List available databases
        if args.list:
            available = list_available_databases()
            print("Available database configurations:")
            for db_type, environments in available.items():
                print(f"  {db_type}: {', '.join(environments)}")
            return 0
        
        # Validate database specification
        if not validate_database_specification(args.database):
            print(f"Invalid database specification: {args.database}")
            print("Format should be: type:environment (e.g., impala:prod, oracle:dev)")
            return 1
        
        # Get credentials
        cred_manager = EnhancedCredentialManager()
        db_parts = args.database.split(':')
        db_type = DatabaseType(db_parts[0])
        environment = Environment(db_parts[1])
        
        user, password = cred_manager.get_credentials(db_type, environment, args.user, args.password)
        
        # Create database client
        manager = create_database_manager()
        
        # Test connection
        if args.test:
            print(f"Testing connection to {args.database}...")
            success = manager.test_connection(db_type, environment, user, password)
            if success:
                print("✅ Connection test successful!")
                return 0
            else:
                print("❌ Connection test failed!")
                return 1
        
        # Execute test query
        if args.query:
            print(f"Executing query on {args.database}: {args.query}")
            client = manager.get_database_client(db_type, environment, user, password)
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.result', delete=False) as temp_file:
                result_file = temp_file.name
            
            try:
                success = client.execute_query(args.query, result_file)
                if success:
                    with open(result_file, 'r') as f:
                        print("Query results:")
                        print(f.read())
                    return 0
                else:
                    print("❌ Query execution failed!")
                    return 1
            finally:
                if os.path.exists(result_file):
                    os.unlink(result_file)
        
        print("No action specified. Use --test or --query options.")
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())