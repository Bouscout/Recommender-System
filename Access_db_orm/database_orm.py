
import pyodbc
from Access_db_orm.database_fetcher import Queries_fetching

class Access_db_ORM :
    def __init__(self, table_name:str, path:str=None) -> None:
        # specify the path to your database
        file_path = r"C:\Users\Ghost\recommender_system\buushido_db.accdb" if not path else path
        
        full_path = (
            r"DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};"
            f"DBQ={file_path}"
        )
        self.full_path = full_path
        self.table = table_name

        self.connect()

        # for fetch operations
        self.objects = Queries_fetching(self.conn, self.table)
        
    def set_table(self, table_name:str):
        self.table = table_name
        self.objects.table = table_name

    def connect(self, path=None):
        path = path if path else self.full_path
        self.conn = pyodbc.connect(path)

        self.cursor = self.conn.cursor()

    def insert(self, entries:dict):
        """
        Insert new values in table from values provided for each columns
        """
        columns_str = ", ".join(list(entries.keys()))
        entry_slot = ", ".join(["?" for _ in entries])

        # Construct the INSERT query
        insert_table = f"INSERT INTO {self.table} ({columns_str}) VALUES ({entry_slot});"

        try :
            self.cursor.execute(insert_table, list(entries.values()))
        except : 
            print(insert_table)
            print("=================")
            print("entries : ", entries)
            self.cursor.rollback()
            raise ValueError(f"value error for this entry")
        
    def update(self, entries:dict, condition:tuple):
        """Update an existing entry"""
        params_str = ",".join([f"{param}=?" for param in list(entries.keys())])

        condition_column, condition_value = condition

        command = f"""
        UPDATE {self.table}
        SET {params_str}
        WHERE {condition_column}=?
        """
        try : 
            self.cursor.execute(command, (*list(entries.values()), condition_value))
        except pyodbc.Error as e:
            print("======================")
            print("error : ", e)
            print("======================")
            print("command : ", command)
            self.cursor.rollback()
            raise ValueError("Something went wrong with those values passed")

    def change_type(self, column:str, new_type:str) :
        """
        Command to change the data type of a column\n
        Ensure that the change internally make sense before trying to apply this
        """
        change_table = f"ALTER TABLE {self.table} ALTER COLUMN {column} {new_type} "

        try :
            self.cursor.execute(change_table)
            print("success")
        except Exception as e: 
            self.cursor.rollback()
            print("error : ", e)

    def create_table(self, table_name, columns_w_types:dict) :
        """
        Create a new table with the provided columns\n

        pass the columns as a list of tuple containing the colum name and data type\n

        ex : >>columns = [(age, INT), (name, TEXT)]

        """
        column_str = ""
        for key, data_type in columns_w_types.items() :
            column_str += f"{key} {data_type},"

        # delete the "," at the end
        column_str = column_str[:-1]

        create_table = f"""
        CREATE TABLE {table_name} (
            ID AUTOINCREMENT PRIMARY KEY,
            {column_str}
        )
        """
        try :
            self.cursor.execute(create_table)
        except pyodbc.Error as e: 
            print("the error is : ", e)
            print("================")
            print("entry : ", create_table)

            
    def delete_table(self, table=None):
        """Delete table provided and if none provided, delete the actual working table"""
        table = table if table else self.table
        command = f"DROP TABLE {table}"
        self.cursor.execute(command)

    def add_column(self, column:str, data_type:str):
        """
        Add a new column to an existing table
        """
        

        command = f"""
        ALTER TABLE {self.table} ADD COLUMN {column} {data_type}
        """
        self.cursor.execute(command)


    def save(self):
        """Commit the changes"""
        self.conn.commit()
        print("changes commited")


    def close(self):
        self.conn.close()


