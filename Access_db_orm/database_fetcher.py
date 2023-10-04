# in order to handle all the fetch
import pandas as pd
import pyodbc

class Queries_fetching:
    def __init__(self, cursor, table) -> None:
        """
        Handle all interactions concerning getting data from the database
        """
        self.cursor = cursor
        self.table = table
        
    def get(self, columns=None, condition=None, range=None, table=None) :
        """
        get all the data for the given column and res
        """
        table = table if table else self.table

        column_str = "*"
        if hasattr(columns, "__len__") :
            column_str = ",".join(columns) 

        select = f"SELECT {column_str} FROM {table}"
        if condition :
            select += f" WHERE {condition}"

        try :
            data = pd.read_sql(select, self.cursor)
            # self.cursor.execute(select)
            # data = self.cursor.fetchall()
            print("success")
            return data
        
        except pyodbc.Error as e :
            self.conn.rollback()
            print("the error : ", e)
            print("===============")
            print("entry : ", select)
        