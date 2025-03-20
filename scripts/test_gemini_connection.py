# test_gemini_connection.py

import os
import google.generativeai as genai
from dotenv import load_dotenv

def test_gemini_connection():
    """
    Simple test to verify connection to Google Gemini API
    """
    print("\n=== Testing Google Gemini API Connection ===\n")
    
    # Load environment variables
    load_dotenv()
    
    # Check for API key
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        api_key = input("Enter your Google AI API key: ").strip()
        if not api_key:
            print("No API key provided. Cannot continue.")
            return
    
    print("API key found. Configuring Gemini...")
    
    try:
        # Configure the API
        genai.configure(api_key=api_key)
        
        # List available models
        print("\nListing available models:")
        models = genai.list_models()
        for model in models:
            if "gemini" in model.name:
                print(f"- {model.name}")
        
        # Test generation with Gemini Pro
        print("\nTesting text generation with Gemini Pro...")
        model = genai.GenerativeModel('gemini-2.0-flash-lite')
        response = model.generate_content("Write a one-sentence description of climate change.")
        
        print("\nTest generation result:")
        print("-" * 50)
        print(response.text)
        print("-" * 50)
        print("✓ Successfully generated text\n")
        
        print("All tests passed! Your Gemini API is configured correctly.")
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        print("Please check your API key and internet connection.")

if __name__ == "__main__":
    test_gemini_connection()