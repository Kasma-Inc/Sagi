#!/usr/bin/env python3
"""
Simple test script to demonstrate the timing functionality in queries.py
"""

import asyncio
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from Sagi.utils.queries import (
    time_function, 
    async_time_operation, 
    time_operation,
    TEST_TIMING
)

print(f"TEST_TIMING is set to: {TEST_TIMING}")

# Test the timing decorators
@time_function("test_sync_function")
def test_sync_function(duration: float):
    """Test synchronous function with timing."""
    import time
    time.sleep(duration)
    return f"Slept for {duration} seconds"

@time_function("test_async_function")
async def test_async_function(duration: float):
    """Test asynchronous function with timing."""
    await asyncio.sleep(duration)
    return f"Async slept for {duration} seconds"

async def test_context_managers():
    """Test the timing context managers."""
    print("\n=== Testing Context Managers ===")
    
    # Test synchronous context manager
    with time_operation("Synchronous operation"):
        import time
        time.sleep(0.1)
        print("Did some sync work")
    
    # Test asynchronous context manager
    async with async_time_operation("Asynchronous operation"):
        await asyncio.sleep(0.1)
        print("Did some async work")

async def main():
    print("=== Testing Timing Functions ===")
    
    # Test synchronous function
    result1 = test_sync_function(0.1)
    print(f"Sync result: {result1}")
    
    # Test asynchronous function
    result2 = await test_async_function(0.1)
    print(f"Async result: {result2}")
    
    # Test context managers
    await test_context_managers()
    
    print("\n=== Testing Nested Operations ===")
    
    # Test nested timing operations
    async with async_time_operation("Outer operation"):
        await asyncio.sleep(0.05)
        
        async with async_time_operation("Inner operation 1"):
            await asyncio.sleep(0.1)
        
        async with async_time_operation("Inner operation 2"):
            await asyncio.sleep(0.1)
        
        await asyncio.sleep(0.05)

if __name__ == "__main__":
    asyncio.run(main())
