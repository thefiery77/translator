import asyncio
from motor.motor_asyncio import AsyncIOMotorClient

async def test():
    uri = "mongodb+srv://translator:dO5fzq5vWpHLHbVt@cluster0.uptrp5y.mongodb.net/?appName=Cluster0&tlsAllowInvalidCertificates=true"
    client = AsyncIOMotorClient(uri, serverSelectionTimeoutMS=5000)
    db = client["translator"]
    books = db["books"]
    try:
        result = await asyncio.wait_for(books.find_one(), timeout=5)
        print(f"✓ Query successful: {result}")
    except asyncio.TimeoutError:
        print("✗ Query timed out after 5s")
    except Exception as e:
        print(f"✗ Error: {type(e).__name__}: {e}")
    finally:
        client.close()

asyncio.run(test())
