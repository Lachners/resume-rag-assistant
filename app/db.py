import psycopg2

conn = psycopg2.connect(
    dbname="jobs",
    user="seblach",
    password="localhost123",
    host="localhost",
)

cur = conn.cursor() # This is a cursor object that allows you to execute SQL queries and fetch results from the database.

cur.execute("SELECT COUNT(*) FROM dataset;") # This executes a SQL query to select all rows from the "database" table.
count = cur.fetchone()[0] # This fetches the first row of the result set and gets the first column (the count of rows)
print(f"Number of rows in the dataset: {count}") # This prints the count of rows in the dataset.

cur.close()
conn.close() # This closes the cursor and the database connection.
