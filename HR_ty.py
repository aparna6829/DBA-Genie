import cx_Oracle
import psycopg2
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Oracle connection details
oracle_dsn = 'promptoraserver:1521/xepdb1'
oracle_user = 'promptora'
oracle_password = '1234'
oracle_schema = 'HR'

# PostgreSQL connection details
pg_dsn = 'dbname=postgres user=postgres password=promptora host=localhost port=5432'
pg_schema = 'HR'  # Already uppercase

# Mapping of Oracle types to PostgreSQL types
type_mapping = {
    'VARCHAR2': 'VARCHAR',
    'NUMBER': 'NUMERIC',
    'DATE': 'DATE',
    'TIMESTAMP': 'TIMESTAMP',
    'CLOB': 'TEXT',
    'BLOB': 'BYTEA',
    'OBJECT': 'JSONB',
    'CHAR': 'CHAR(255)'
}

def fetch_oracle_types(oracle_schema):
    try:
        connection = cx_Oracle.connect(user=oracle_user, password=oracle_password, dsn=oracle_dsn)
        cursor = connection.cursor()
        
        # Fetch types
        cursor.execute(f"""
            SELECT 
                UPPER(type_name),
                typecode
            FROM
                ALL_TYPES
            WHERE 
                OWNER = :schema
        """, schema=oracle_schema)
        types = cursor.fetchall()
        logging.debug(f"Fetched types from Oracle: {types}")
        
        # Fetch attributes for OBJECT types
        object_types = [t[0] for t in types if t[1] == 'OBJECT']
        type_details = {}
        for obj_type in object_types:
            cursor.execute(f"""
                SELECT 
                    UPPER(ATTR_NAME), 
                    UPPER(ATTR_TYPE_NAME)
                FROM 
                    ALL_TYPE_ATTRS
                WHERE 
                    OWNER = :schema AND
                    TYPE_NAME = :type_name
            """, schema=oracle_schema, type_name=obj_type)
            attributes = cursor.fetchall()
            type_details[obj_type] = attributes
            logging.debug(f"Attributes for {obj_type}: {attributes}")
        
        return types, type_details
    except cx_Oracle.Error as error:
        logging.error(f"Error fetching Oracle types: {error}")
        raise
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

def create_pg_types(types, type_details):
    connection = None
    cursor = None
    created_types = 0
    try:
        connection = psycopg2.connect(dsn=pg_dsn)
        cursor = connection.cursor()

        # Create schema if it doesn't exist
        cursor.execute(f'CREATE SCHEMA IF NOT EXISTS "{pg_schema}"')
        logging.debug(f'Ensured schema "{pg_schema}" exists')

        for type_name, oracle_type in types:
            if oracle_type == 'OBJECT':
                attributes = type_details.get(type_name, [])
                pg_attributes = []
                for attr_name, attr_type in attributes:
                    pg_type = type_mapping.get(attr_type, 'TEXT')
                    pg_attributes.append(f'"{attr_name}" {pg_type}')
                
                pg_type_definition = ', '.join(pg_attributes)
                create_statement = f"""
                    CREATE TYPE "{pg_schema}"."{type_name}" AS (
                        {pg_type_definition}
                    );
                """
            else:
                pg_type = type_mapping.get(oracle_type.upper())
                if not pg_type:
                    logging.warning(f"No mapping found for Oracle type: {oracle_type}")
                    continue
                create_statement = f"""
                    CREATE TYPE "{pg_schema}"."{type_name}" AS ({pg_type});
                """
            
            try:
                cursor.execute(create_statement)
                logging.debug(f"Created PostgreSQL type: {type_name}")
                created_types += 1
            except psycopg2.Error as error:
                logging.warning(f"Error creating type {type_name}: {error}")

        connection.commit()
        return created_types
    except psycopg2.Error as error:
        logging.error(f"Error in PostgreSQL operation: {error}")
        if connection:
            connection.rollback()
        raise
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

def main():
    try:
        types, type_details = fetch_oracle_types(oracle_schema)
        created_types = create_pg_types(types, type_details)
        if created_types > 0:
            logging.info(f"Type migration completed successfully. Created {created_types} types.")
        else:
            logging.warning("No types were migrated.")
    except Exception as error:
        logging.error(f"An error occurred during migration: {error}")

if __name__ == '__main__':
    main()