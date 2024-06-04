from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import WebDriverException
import time
import base64
from PIL import Image
import pytesseract
import imagehash
from sklearn.feature_extraction.text import TfidfVectorizer
import subprocess

# Initially vectorise using tfidr and then check similarity using cosine similarity
def calculate_text_similarity(paragraph1, paragraph2):
    if not paragraph1 or not paragraph2:
        return 0.0
    
    sentences = [paragraph1, paragraph2]
    vectorizer = TfidfVectorizer()
    
    try:
        tfidf_matrix = vectorizer.fit_transform(sentences)
        cosine_sim = (tfidf_matrix * tfidf_matrix.T).A
        similarity_percentage = cosine_sim[0, 1] * 100
    except ValueError as e:
        print(f"Error calculating similarity: {e}")
        similarity_percentage = 0.0
    
    return similarity_percentage

def extract_text(image_path):
    # Use Tesseract to extract text from the image
    text = pytesseract.image_to_string(Image.open(image_path))
    return text

def calculate_image_hash(image_path):
    # Calculate the perceptual hash (phash) of the image
    hash_value = imagehash.phash(Image.open(image_path))
    return hash_value

def take_full_page_screenshot(url, file_name, retries=3):
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--ignore-certificate-errors')
    options.add_argument('--disable-dev-shm-usage')

    while retries > 0:
        try:
            driver = webdriver.Chrome(options=options)
            driver.set_window_size(1920, 1080)
            driver.get(url)
            
            # Use WebDriverWait to wait until the page is fully loaded
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, 'body'))
            )
            
            # Get the dimensions of the page
            total_width = driver.execute_script("return document.body.scrollWidth")
            total_height = driver.execute_script("return document.body.scrollHeight")
            driver.set_window_size(total_width, total_height)
            time.sleep(2)  # Adjust the sleep time if necessary
            
            # Take the screenshot
            screenshot_base64 = driver.get_screenshot_as_base64()
            with open(file_name, "wb") as f:
                f.write(base64.b64decode(screenshot_base64))
            
            driver.quit()
            return

        except WebDriverException as e:
            print(f"Error occurred: {e}")
            retries -= 1
            driver.quit()
            if retries == 0:
                print("Max retries reached. Unable to take screenshot.")
                raise

def compare_text(text1, text2):
    return calculate_text_similarity(text1, text2)

def calculate_image_similarity(image1_path, image2_path):
    try:
        result = subprocess.run(['magick', 'compare', '-metric', 'RMSE', image1_path, image2_path, 'null:'], capture_output=True, text=True)
        output = result.stderr.strip() if result.stderr else result.stdout.strip()
        
        if "Error:" in output:
            print(output)
            similarity_score = 0.0
            normalized_similarity_score = 0.0
        else:
            similarity_score = float(output.split()[0])
            normalized_similarity_score = float(output.split()[1][1:-1])  # Extracting normalized RMSE
        
    except Exception as e:
        print(f"Error calculating image similarity: {e}")
        similarity_score = 0.0
        normalized_similarity_score = 0.0
    
    return similarity_score, normalized_similarity_score
# Check for Domain based analysis.
def main():
    target_url = 'https://www.amazon.com'
    newly_registered_url = 'https://qqq.sgdf45trt.cloudns.be/index.html'
    # Paths to the screenshots
    screenshot1_path = 'screenshot1.png'
    screenshot2_path = 'screenshot2.png'
    text_output1_path = 'screenshot1.txt'
    text_output2_path = 'screenshot2.txt'

    take_full_page_screenshot(target_url, screenshot1_path)
    take_full_page_screenshot(newly_registered_url, screenshot2_path)

    # Extract text from the screenshots
    text1 = extract_text(screenshot1_path)
    text2 = extract_text(screenshot2_path)

    # Write extracted text into files
    with open(text_output1_path, 'w', encoding='utf-8') as file1:
        file1.write(text1)
    with open(text_output2_path, 'w', encoding='utf-8') as file2:
        file2.write(text2)

    # Calculate perceptual hashes of the screenshots
    hash1 = calculate_image_hash(screenshot1_path)
    hash2 = calculate_image_hash(screenshot2_path)

     # Compare text similarity
    text_similarity = compare_text(text1, text2)
    print(f'Text Similarity Score: {text_similarity:.2f}')

    # Compare image similarity
    image_similarity, normalized_image_similarity = calculate_image_similarity(screenshot1_path, screenshot2_path)
    print(f'Image Similarity Score (RMSE): {image_similarity:.2f}')
    print(f'Image Similarity Score (Normalized RMSE): {normalized_image_similarity:.2f}')


if __name__ == '__main__':
    main()