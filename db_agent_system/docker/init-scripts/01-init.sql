-- Create sample tables for a simple e-commerce database

-- Create Products Table
CREATE TABLE IF NOT EXISTS products (
    product_id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    price DECIMAL(10, 2) NOT NULL,
    category VARCHAR(50),
    stock_quantity INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create Customers Table
CREATE TABLE IF NOT EXISTS customers (
    customer_id SERIAL PRIMARY KEY,
    first_name VARCHAR(50) NOT NULL,
    last_name VARCHAR(50) NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    phone VARCHAR(20),
    address TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create Orders Table
CREATE TABLE IF NOT EXISTS orders (
    order_id SERIAL PRIMARY KEY,
    customer_id INTEGER REFERENCES customers(customer_id),
    order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    total_amount DECIMAL(12, 2) NOT NULL,
    status VARCHAR(20) DEFAULT 'pending',
    shipping_address TEXT,
    payment_method VARCHAR(50)
);

-- Create Order_Items Table (junction table between Orders and Products)
CREATE TABLE IF NOT EXISTS order_items (
    order_item_id SERIAL PRIMARY KEY,
    order_id INTEGER REFERENCES orders(order_id),
    product_id INTEGER REFERENCES products(product_id),
    quantity INTEGER NOT NULL,
    unit_price DECIMAL(10, 2) NOT NULL,
    subtotal DECIMAL(10, 2) NOT NULL
);

-- Insert sample data into Products
INSERT INTO products (name, description, price, category, stock_quantity)
VALUES
    ('Laptop Pro', 'High-performance laptop for professionals', 1299.99, 'Electronics', 50),
    ('Smartphone X', 'Latest smartphone with advanced features', 799.99, 'Electronics', 100),
    ('Coffee Maker', 'Automatic coffee maker for home use', 79.99, 'Home Appliances', 30),
    ('Running Shoes', 'Comfortable shoes for running and training', 89.99, 'Sportswear', 75),
    ('Desk Lamp', 'Adjustable LED desk lamp', 29.99, 'Home Accessories', 120);

-- Insert sample data into Customers
INSERT INTO customers (first_name, last_name, email, phone, address)
VALUES
    ('John', 'Doe', 'john.doe@example.com', '555-123-4567', '123 Main St, Anytown, USA'),
    ('Jane', 'Smith', 'jane.smith@example.com', '555-987-6543', '456 Oak St, Somewhere, USA'),
    ('Bob', 'Johnson', 'bob.johnson@example.com', '555-567-8901', '789 Pine St, Nowhere, USA'),
    ('Alice', 'Williams', 'alice.williams@example.com', '555-345-6789', '321 Elm St, Anywhere, USA'),
    ('Charlie', 'Brown', 'charlie.brown@example.com', '555-678-9012', '654 Maple St, Everywhere, USA');

-- Insert sample data into Orders
INSERT INTO orders (customer_id, total_amount, status, shipping_address, payment_method)
VALUES
    (1, 1299.99, 'completed', '123 Main St, Anytown, USA', 'Credit Card'),
    (2, 79.99, 'processing', '456 Oak St, Somewhere, USA', 'PayPal'),
    (3, 89.99, 'shipped', '789 Pine St, Nowhere, USA', 'Credit Card'),
    (4, 829.98, 'completed', '321 Elm St, Anywhere, USA', 'Debit Card'),
    (5, 29.99, 'pending', '654 Maple St, Everywhere, USA', 'PayPal');

-- Insert sample data into Order_Items
INSERT INTO order_items (order_id, product_id, quantity, unit_price, subtotal)
VALUES
    (1, 1, 1, 1299.99, 1299.99),
    (2, 3, 1, 79.99, 79.99),
    (3, 4, 1, 89.99, 89.99),
    (4, 2, 1, 799.99, 799.99),
    (4, 3, 1, 29.99, 29.99),
    (5, 5, 1, 29.99, 29.99); 