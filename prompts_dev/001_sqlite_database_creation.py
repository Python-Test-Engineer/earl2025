import sqlite3
import random
from datetime import datetime, timedelta
import uuid

# Create a connection to the SQLite database
conn = sqlite3.connect('retail_store.db')
cursor = conn.cursor()

# Enable foreign keys
cursor.execute('PRAGMA foreign_keys = ON;')

# Create tables with appropriate constraints and relationships
# 1. Categories Table
cursor.execute('''
CREATE TABLE IF NOT EXISTS categories (
    category_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    active BOOLEAN DEFAULT TRUE CHECK (active IN (0, 1))
);
''')

# 2. Customers Table
cursor.execute('''
CREATE TABLE IF NOT EXISTS customers (
    customer_id INTEGER PRIMARY KEY,
    first_name TEXT NOT NULL,
    last_name TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE,
    phone TEXT,
    address TEXT,
    city TEXT,
    postcode TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_purchase_date TIMESTAMP,
    loyalty_points INTEGER DEFAULT 0 CHECK (loyalty_points >= 0)
);
''')

# 3. Products Table
cursor.execute('''
CREATE TABLE IF NOT EXISTS products (
    product_id INTEGER PRIMARY KEY,
    category_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    price DECIMAL(10, 2) NOT NULL CHECK (price >= 0),
    stock_quantity INTEGER NOT NULL DEFAULT 0 CHECK (stock_quantity >= 0),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    active BOOLEAN DEFAULT TRUE CHECK (active IN (0, 1)),
    sku TEXT UNIQUE,
    FOREIGN KEY (category_id) REFERENCES categories(category_id) ON DELETE RESTRICT
);
''')

# 4. Orders Table
cursor.execute('''
CREATE TABLE IF NOT EXISTS orders (
    order_id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL,
    order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status TEXT NOT NULL CHECK (status IN ('pending', 'processing', 'shipped', 'delivered', 'cancelled')),
    shipping_address TEXT,
    shipping_city TEXT,
    shipping_postcode TEXT,
    total_amount DECIMAL(10, 2) NOT NULL CHECK (total_amount >= 0),
    payment_method TEXT CHECK (payment_method IN ('credit_card', 'debit_card', 'paypal', 'bank_transfer')),
    notes TEXT,
    tracking_number TEXT,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id) ON DELETE RESTRICT
);
''')

# 5. Order Items Table (junction table for orders and products)
cursor.execute('''
CREATE TABLE IF NOT EXISTS order_items (
    order_item_id INTEGER PRIMARY KEY,
    order_id INTEGER NOT NULL,
    product_id INTEGER NOT NULL,
    quantity INTEGER NOT NULL CHECK (quantity > 0),
    unit_price DECIMAL(10, 2) NOT NULL CHECK (unit_price >= 0),
    subtotal DECIMAL(10, 2) NOT NULL CHECK (subtotal >= 0),
    FOREIGN KEY (order_id) REFERENCES orders(order_id) ON DELETE CASCADE,
    FOREIGN KEY (product_id) REFERENCES products(product_id) ON DELETE RESTRICT,
    UNIQUE(order_id, product_id)
);
''')

# Sample data for categories
categories = [
    (1, 'Electronics', 'Electronic devices and accessories'),
    (2, 'Clothing', 'Apparel and fashion items'),
    (3, 'Home & Kitchen', 'Household items and kitchenware'),
    (4, 'Books', 'Books, e-books, and publications'),
    (5, 'Sports & Outdoors', 'Sporting goods and outdoor equipment'),
    (6, 'Beauty & Personal Care', 'Cosmetics and personal hygiene products'),
    (7, 'Toys & Games', 'Entertainment items for children and adults'),
    (8, 'Grocery', 'Food and beverage items'),
    (9, 'Health & Wellness', 'Health supplements and medical equipment'),
    (10, 'Office Supplies', 'Stationery and office equipment')
]

# Insert categories data
cursor.executemany('INSERT INTO categories (category_id, name, description) VALUES (?, ?, ?)', categories)

# Sample data for customers
customers = [
    (1, 'John', 'Smith', 'john.smith@email.com', '07700123456', '123 High Street', 'London', 'SW1A 1AA', '2023-01-15', 450),
    (2, 'Emma', 'Jones', 'emma.jones@email.com', '07700234567', '456 Park Lane', 'Manchester', 'M1 1AA', '2023-02-20', 200),
    (3, 'Michael', 'Brown', 'michael.brown@email.com', '07700345678', '789 Oak Road', 'Birmingham', 'B1 1AA', '2023-03-10', 300),
    (4, 'Sarah', 'Wilson', 'sarah.wilson@email.com', '07700456789', '101 Pine Avenue', 'Glasgow', 'G1 1AA', '2023-03-25', 150),
    (5, 'David', 'Taylor', 'david.taylor@email.com', '07700567890', '202 Elm Street', 'Liverpool', 'L1 1AA', '2023-04-05', 500),
    (6, 'Lisa', 'Evans', 'lisa.evans@email.com', '07700678901', '303 Maple Drive', 'Bristol', 'BS1 1AA', '2023-04-20', 250),
    (7, 'James', 'Thomas', 'james.thomas@email.com', '07700789012', '404 Cedar Lane', 'Newcastle', 'NE1 1AA', '2023-05-10', 350),
    (8, 'Emily', 'Roberts', 'emily.roberts@email.com', '07700890123', '505 Birch Road', 'Leeds', 'LS1 1AA', '2023-05-25', 100),
    (9, 'Robert', 'Johnson', 'robert.johnson@email.com', '07700901234', '606 Willow Street', 'Sheffield', 'S1 1AA', '2023-06-15', 550),
    (10, 'Jessica', 'Walker', 'jessica.walker@email.com', '07700012345', '707 Fir Avenue', 'Edinburgh', 'EH1 1AA', '2023-07-01', 275)
]

# Insert customers data
cursor.executemany('''
INSERT INTO customers 
(customer_id, first_name, last_name, email, phone, address, city, postcode, last_purchase_date, loyalty_points) 
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
''', customers)

# Sample data for products
products = [
    (1, 1, 'Smartphone X1', 'Latest smartphone with advanced features', 899.99, 50, '2023-01-10', 'ELEC-001'),
    (2, 1, 'Wireless Headphones', 'Noise-cancelling bluetooth headphones', 129.99, 100, '2023-01-15', 'ELEC-002'),
    (3, 1, 'Smart Watch', 'Fitness and health tracking watch', 199.99, 75, '2023-01-20', 'ELEC-003'),
    (4, 2, 'Cotton T-Shirt', 'Comfortable cotton t-shirt for daily wear', 19.99, 200, '2023-02-01', 'CLTH-001'),
    (5, 2, 'Denim Jeans', 'Classic blue denim jeans', 49.99, 150, '2023-02-05', 'CLTH-002'),
    (6, 3, 'Coffee Maker', 'Automatic coffee brewing machine', 89.99, 40, '2023-02-10', 'HOME-001'),
    (7, 3, 'Non-stick Pan Set', '3-piece non-stick cookware set', 59.99, 60, '2023-02-15', 'HOME-002'),
    (8, 4, 'Mystery Novel', 'Best-selling mystery thriller book', 12.99, 120, '2023-03-01', 'BOOK-001'),
    (9, 4, 'Cookbook', 'Collection of gourmet recipes', 24.99, 80, '2023-03-05', 'BOOK-002'),
    (10, 5, 'Yoga Mat', 'Non-slip exercise yoga mat', 29.99, 100, '2023-03-10', 'SPRT-001'),
    (11, 5, 'Dumbbells Set', '5kg pair of fitness dumbbells', 45.99, 60, '2023-03-15', 'SPRT-002'),
    (12, 6, 'Facial Cleanser', 'Gentle facial cleansing gel', 15.99, 90, '2023-04-01', 'BEAU-001'),
    (13, 6, 'Moisturizer', 'Hydrating face and body moisturizer', 22.99, 85, '2023-04-05', 'BEAU-002'),
    (14, 7, 'Board Game', 'Family strategy board game', 34.99, 70, '2023-04-10', 'TOYS-001'),
    (15, 7, 'Puzzle Set', '1000-piece landscape jigsaw puzzle', 19.99, 65, '2023-04-15', 'TOYS-002'),
    (16, 8, 'Organic Tea', 'Assorted organic tea varieties', 8.99, 150, '2023-05-01', 'GROC-001'),
    (17, 8, 'Chocolate Box', 'Assorted premium chocolate box', 14.99, 100, '2023-05-05', 'GROC-002'),
    (18, 9, 'Vitamin Supplements', 'Daily multivitamin tablets', 17.99, 120, '2023-05-10', 'HLTH-001'),
    (19, 9, 'Digital Thermometer', 'Fast-reading digital thermometer', 12.99, 80, '2023-05-15', 'HLTH-002'),
    (20, 10, 'Notebook Set', '3-pack hardcover ruled notebooks', 9.99, 200, '2023-06-01', 'OFFC-001')
]

# Insert products data
cursor.executemany('''
INSERT INTO products 
(product_id, category_id, name, description, price, stock_quantity, created_at, sku) 
VALUES (?, ?, ?, ?, ?, ?, ?, ?)
''', products)

# Generate sample orders and order items
order_statuses = ['pending', 'processing', 'shipped', 'delivered', 'cancelled']
payment_methods = ['credit_card', 'debit_card', 'paypal', 'bank_transfer']

# Generate 50 random orders
orders = []
order_items_list = []
order_item_id = 1

start_date = datetime(2023, 1, 1)
end_date = datetime(2023, 12, 31)

for order_id in range(1, 51):
    # Select a random customer
    customer_id = random.randint(1, 10)
    
    # Get customer info for shipping
    cursor.execute('SELECT address, city, postcode FROM customers WHERE customer_id = ?', (customer_id,))
    customer_info = cursor.fetchone()
    
    # Generate a random date between start_date and end_date
    days_between = (end_date - start_date).days
    random_days = random.randint(0, days_between)
    order_date = start_date + timedelta(days=random_days)
    
    # Select random status and payment method
    status = random.choice(order_statuses)
    payment_method = random.choice(payment_methods)
    
    # Generate a tracking number for shipped or delivered orders
    tracking_number = uuid.uuid4().hex[:12].upper() if status in ['shipped', 'delivered'] else None
    
    # Calculate total amount based on order items (will be updated later)
    total_amount = 0.0
    
    # Add order to list
    orders.append((
        order_id,
        customer_id,
        order_date.strftime('%Y-%m-%d %H:%M:%S'),
        status,
        customer_info[0],  # address
        customer_info[1],  # city
        customer_info[2],  # postcode
        total_amount,  # Placeholder, will update after order items
        payment_method,
        f"Order #{order_id} notes",
        tracking_number
    ))
    
    # Generate 1-5 random order items for this order
    num_items = random.randint(1, 5)
    
    # Keep track of products already added to this order to avoid duplicates
    added_products = set()
    
    order_total = 0.0
    
    for _ in range(num_items):
        # Select a random product not already in this order
        while True:
            product_id = random.randint(1, 20)
            if product_id not in added_products:
                added_products.add(product_id)
                break
        
        # Get product price
        cursor.execute('SELECT price FROM products WHERE product_id = ?', (product_id,))
        unit_price = cursor.fetchone()[0]
        
        # Generate random quantity
        quantity = random.randint(1, 5)
        
        # Calculate subtotal
        subtotal = unit_price * quantity
        
        # Add to order total
        order_total += subtotal
        
        # Add order item to list
        order_items_list.append((
            order_item_id,
            order_id,
            product_id,
            quantity,
            unit_price,
            subtotal
        ))
        
        order_item_id += 1
    
    # Update total amount in orders
    orders[order_id-1] = orders[order_id-1][:7] + (order_total,) + orders[order_id-1][8:]

# Insert orders data
cursor.executemany('''
INSERT INTO orders 
(order_id, customer_id, order_date, status, shipping_address, shipping_city, shipping_postcode, 
 total_amount, payment_method, notes, tracking_number) 
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
''', orders)

# Insert order items data
cursor.executemany('''
INSERT INTO order_items 
(order_item_id, order_id, product_id, quantity, unit_price, subtotal) 
VALUES (?, ?, ?, ?, ?, ?)
''', order_items_list)

# Commit the changes and close the connection
conn.commit()

# Verify the data was inserted correctly
def count_records(table_name):
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    return cursor.fetchone()[0]

print(f"Categories: {count_records('categories')} records")
print(f"Customers: {count_records('customers')} records")
print(f"Products: {count_records('products')} records")
print(f"Orders: {count_records('orders')} records")
print(f"Order Items: {count_records('order_items')} records")

conn.close()
