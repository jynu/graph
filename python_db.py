How to Replace the Java Client:
1. Update DatabaseClient class in training_data_main.py:
Replace the execute_query method:
pythonimport sqlalchemy
from sqlalchemy import create_engine, text
import pandas as pd

class DatabaseClient:
    def __init__(self, db_config: DatabaseConfig, system_config: SystemConfig):
        self.db_config = db_config
        self.system_config = system_config
        self.logger = logging.getLogger(__name__)
        self.engine = None
        self._create_connection()
    
    def _create_connection(self):
        """Create database connection"""
        try:
            # Example connection string - adjust for your database
            # For SQL Server:
            connection_string = f"mssql+pyodbc://{self.db_config.user_id}:{self.db_config.password}@{self.db_config.host}/your_database?driver=ODBC+Driver+17+for+SQL+Server"
            
            # For PostgreSQL:
            # connection_string = f"postgresql://{self.db_config.user_id}:{self.db_config.password}@{self.db_config.host}/your_database"
            
            self.engine = create_engine(connection_string)
            self.logger.info("Database connection created successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to create database connection: {e}")
            raise
    
    def execute_query(self, query: str, output_file: str, format_type: str = "TXT") -> bool:
        """Execute a database query and save results to file"""
        try:
            self.logger.info(f"Executing query: {query}")
            
            # Execute query using pandas for easy file output
            df = pd.read_sql(query, self.engine)
            
            # Save to file based on format
            if format_type.upper() == "JSON":
                df.to_json(output_file, orient='records', indent=2)
            else:  # TXT/CSV format
                df.to_csv(output_file, index=False, sep=',')
            
            self.logger.info(f"Query results saved to: {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error executing query: {e}")
            return False
    
    def test_connection(self) -> bool:
        """Test database connection"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1 as test"))
                test_value = result.fetchone()[0]
                self.logger.info(f"Connection test successful: {test_value}")
                return True
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            return False
2. Update ConfigManager._get_host_from_config():
pythondef _get_host_from_config(self, config_data: dict) -> str:
    """Extract host configuration"""
    # Instead of the WebSocket-style host, use regular database host
    return config_data.get('HOST', 'your-database-server.com')
3. Update your config file format:
ini[transcation]
TABLES = table1,table2
HOST = your-database-server.com
DATABASE = your_database_name
USERID = your_username
PASSWORD = your_password
BASEDIRECTORY = ./output
OUTPUT_DIRECTORY = ./final_output
START_DATE = 20250101

# Optional: Specify database type
DB_TYPE = sqlserver  # or postgresql, oracle, mysql
4. Add requirements.txt: