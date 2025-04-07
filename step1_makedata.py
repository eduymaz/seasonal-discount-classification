from sqlalchemy import create_engine
import pandas as pd

DB_USERNAME = "postgres"
DB_PASSWORD = "123456"
DB_HOST = "localhost" 
DB_PORT = "5432"
DB_NAME = "Gyk1Nortwind"

db_url = f"postgresql://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(db_url)


query = '''

SELECT
    p.product_id,
    p.product_name,
    p.category_id,
    c.category_name,
    SUM(od.quantity) AS total_quantity,
    SUM(od.unit_price * od.quantity * (1 - od.discount)) AS total_sales,
    AVG(od.unit_price) AS avg_price
FROM
    order_details od
LEFT JOIN
    products p ON od.product_id = p.product_id
LEFT JOIN
    categories c ON p.category_id = c.category_id
GROUP BY
    p.product_id, p.product_name, p.category_id, c.category_name
ORDER BY
    total_sales DESC;'''

df = pd.read_sql(query, engine)
#df.to_csv("./data/data.csv", index=False)