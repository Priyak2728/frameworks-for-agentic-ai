from langchain.agents import create_agent
from langchain.tools import tool
from langchain_aws import ChatBedrockConverse


# =========================================================
# 1. BEDROCK MODEL
# =========================================================

model = ChatBedrockConverse(
    model="amazon.nova-lite-v1:0",
    region_name="us-east-1"
)


# =========================================================
# 2. CUSTOM TOOL - ORDER STATUS
# =========================================================

@tool
def get_order_status(order_id: str) -> str:
    """Get the current status of a customer order."""

    orders = {
        "ORD1001": "Shipped",
        "ORD1002": "Delivered",
        "ORD1003": "Processing",
        "ORD1004": "Cancelled"
    }

    return orders.get(
        order_id,
        f"No order found with ID {order_id}"
    )


# =========================================================
# 3. CUSTOM TOOL - PRODUCT PRICE
# =========================================================

@tool
def get_product_price(product: str) -> str:
    """Get the price of a product."""

    products = {
        "laptop": 75000,
        "phone": 45000,
        "headphones": 5000,
        "tablet": 30000
    }

    price = products.get(product.lower())

    if price:
        return f"The price of the {product} is ₹{price}."

    return f"Product '{product}' was not found."


# =========================================================
# 4. CUSTOM TOOL - DISCOUNT
# =========================================================

@tool
def calculate_discount(price: float, percentage: float) -> float:
    """Calculate the final price after applying a discount."""

    discount = price * (percentage / 100)
    final_price = price - discount

    return final_price


# =========================================================
# 5. CREATE AGENT
# =========================================================

agent = create_agent(
    model=model,
    tools=[
        get_order_status,
        get_product_price,
        calculate_discount
    ]
)


# =========================================================
# 6. DYNAMIC USER INPUT
# =========================================================

user_query = input("\nAsk me something: ")


# =========================================================
# 7. INVOKE AGENT
# =========================================================

response = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": user_query
        }
    ]
})


# =========================================================
# 8. GET FINAL ANSWER
# =========================================================

final_message = response["messages"][-1]

print("\nAgent:")

content = final_message.content

if isinstance(content, str):
    print(content)

elif isinstance(content, list):

    for block in content:

        if isinstance(block, dict) and block.get("type") == "text":
            print(block.get("text", ""))