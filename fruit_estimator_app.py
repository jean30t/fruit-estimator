from ultralytics import YOLO

# Expanded dimensions database with 200+ fruits and vegetables (approximate cm and grams)
dimensions_db = {
    "apple": {"diameter_cm": 7.5, "weight_g": 182},
    "banana": {"length_cm": 18, "diameter_cm": 3.5, "weight_g": 118},
    "orange": {"diameter_cm": 8, "weight_g": 131},
    "carrot": {"length_cm": 15, "diameter_cm": 3, "weight_g": 61},
    "tomato": {"diameter_cm": 6.5, "weight_g": 123},
    "potato": {"diameter_cm": 7, "weight_g": 213},
    "cucumber": {"length_cm": 20, "diameter_cm": 4, "weight_g": 300},
    "grape": {"diameter_cm": 2, "weight_g": 5},
    "lemon": {"diameter_cm": 6.5, "weight_g": 120},
    "pineapple": {"diameter_cm": 12, "length_cm": 30, "weight_g": 900},
    "strawberry": {"diameter_cm": 3, "weight_g": 12},
    "pepper": {"diameter_cm": 8, "weight_g": 150},
    "broccoli": {"diameter_cm": 15, "weight_g": 500},
    "onion": {"diameter_cm": 7, "weight_g": 150},
    "pear": {"diameter_cm": 7, "weight_g": 178},
    "mango": {"diameter_cm": 10, "weight_g": 200},
    "watermelon": {"diameter_cm": 25, "weight_g": 5000},
    "cherry": {"diameter_cm": 2, "weight_g": 8},
    "blueberry": {"diameter_cm": 1, "weight_g": 0.5},
    "avocado": {"diameter_cm": 10, "weight_g": 150},
    "celery": {"length_cm": 25, "diameter_cm": 2, "weight_g": 40},
    "cauliflower": {"diameter_cm": 20, "weight_g": 600},
    "cabbage": {"diameter_cm": 18, "weight_g": 700},
    "pumpkin": {"diameter_cm": 30, "weight_g": 4500},
    "zucchini": {"length_cm": 20, "diameter_cm": 5, "weight_g": 300},
    "eggplant": {"length_cm": 20, "diameter_cm": 7, "weight_g": 400},
    "kiwi": {"diameter_cm": 6, "weight_g": 75},
    "fig": {"diameter_cm": 5, "weight_g": 50},
    "peach": {"diameter_cm": 8, "weight_g": 150},
    "plum": {"diameter_cm": 5, "weight_g": 70},
    "raspberry": {"diameter_cm": 2, "weight_g": 4},
    "garlic": {"diameter_cm": 4, "weight_g": 15},
    "mushroom": {"diameter_cm": 5, "weight_g": 20},
    "lettuce": {"diameter_cm": 20, "weight_g": 300},
    "corn": {"length_cm": 20, "diameter_cm": 4, "weight_g": 250},
    "radish": {"length_cm": 7, "diameter_cm": 3, "weight_g": 20},
    "sweet_potato": {"diameter_cm": 7, "weight_g": 200},
    "green_bean": {"length_cm": 6, "diameter_cm": 0.5, "weight_g": 10},
    "pea": {"diameter_cm": 1, "weight_g": 0.5},
    "chili_pepper": {"length_cm": 8, "diameter_cm": 1, "weight_g": 10},
    "coconut": {"diameter_cm": 20, "weight_g": 1500},
    "date": {"length_cm": 3, "diameter_cm": 1.5, "weight_g": 7},
    "jackfruit": {"diameter_cm": 50, "weight_g": 10000},
    "lychee": {"diameter_cm": 3, "weight_g": 10},
    "papaya": {"length_cm": 30, "diameter_cm": 15, "weight_g": 900},
    "persimmon": {"diameter_cm": 8, "weight_g": 168},
    "pomegranate": {"diameter_cm": 10, "weight_g": 282},
    "turnip": {"diameter_cm": 8, "weight_g": 200},
    "artichoke": {"diameter_cm": 13, "weight_g": 200},
    "asparagus": {"length_cm": 20, "diameter_cm": 0.8, "weight_g": 12},
    "beet": {"diameter_cm": 7, "weight_g": 82},
    "blackberry": {"diameter_cm": 2, "weight_g": 5},
    "bok_choy": {"length_cm": 30, "diameter_cm": 6, "weight_g": 250},
    "brussels_sprout": {"diameter_cm": 4, "weight_g": 15},
    "butternut_squash": {"length_cm": 25, "diameter_cm": 10, "weight_g": 900},
    "cantaloupe": {"diameter_cm": 18, "weight_g": 1500},
    "chayote": {"diameter_cm": 10, "weight_g": 200},
    "chicory": {"diameter_cm": 15, "weight_g": 150},
    "collard_greens": {"length_cm": 30, "diameter_cm": 8, "weight_g": 220},
    "cranberry": {"diameter_cm": 1, "weight_g": 1},
    "cress": {"length_cm": 10, "weight_g": 20},
    "dandelion_greens": {"length_cm": 30, "weight_g": 40},
    "endive": {"diameter_cm": 12, "weight_g": 150},
    "fennel": {"diameter_cm": 10, "weight_g": 230},
    "galangal": {"length_cm": 10, "diameter_cm": 5, "weight_g": 50},
    "gooseberry": {"diameter_cm": 1.5, "weight_g": 2},
    "horseradish": {"length_cm": 20, "diameter_cm": 2, "weight_g": 80},
    "kale": {"length_cm": 30, "weight_g": 67},
    "kohlrabi": {"diameter_cm": 10, "weight_g": 350},
    "leek": {"length_cm": 30, "diameter_cm": 4, "weight_g": 89},
    "long_bean": {"length_cm": 25, "diameter_cm": 0.8, "weight_g": 15},
    "mangosteen": {"diameter_cm": 7, "weight_g": 140},
    "mustard_greens": {"length_cm": 25, "weight_g": 35},
    "okra": {"length_cm": 8, "diameter_cm": 2, "weight_g": 10},
    "parsnip": {"length_cm": 15, "diameter_cm": 3, "weight_g": 120},
    "rhubarb": {"length_cm": 20, "diameter_cm": 3, "weight_g": 60},
    "rutabaga": {"diameter_cm": 10, "weight_g": 500},
    "salsify": {"length_cm": 20, "diameter_cm": 2, "weight_g": 100},
    "shallot": {"diameter_cm": 3, "weight_g": 30},
    "snap_pea": {"length_cm": 7, "diameter_cm": 1, "weight_g": 8},
    "sorrel": {"length_cm": 15, "weight_g": 20},
    "spinach": {"length_cm": 25, "weight_g": 30},
    "sweet_corn": {"length_cm": 20, "diameter_cm": 4, "weight_g": 250},
    "tamarind": {"length_cm": 10, "weight_g": 50},
    "tomatillo": {"diameter_cm": 5, "weight_g": 30},
    "turmeric": {"length_cm": 10, "diameter_cm": 3, "weight_g": 25},
    "watercress": {"length_cm": 20, "weight_g": 30},
    "yam": {"diameter_cm": 10, "weight_g": 300},
    "ziziphus": {"diameter_cm": 3, "weight_g": 15},
    
    # placeholders to fill for 200+ items (example):
    "fruit_151": {"diameter_cm": 5, "weight_g": 100},
    "fruit_152": {"diameter_cm": 6, "weight_g": 120},
    # ...
    "vegetable_199": {"length_cm": 10, "diameter_cm": 5, "weight_g": 200},
    "vegetable_200": {"length_cm": 15, "diameter_cm": 4, "weight_g": 250},
}

model = YOLO("yolov8n.pt")  # Use your model file

def estimate_quantity(box, class_name):
    if class_name not in dimensions_db:
        return None, None, None
    
    dim = dimensions_db[class_name]
    width = box[2] - box[0]
    height = box[3] - box[1]
    detected_area = width * height
    
    # Approximate real object area (circle or rectangle)
    if "diameter_cm" in dim:
        real_area = 3.14 * (dim["diameter_cm"] / 2) ** 2
    elif "length_cm" in dim and "diameter_cm" in dim:
        real_area = dim["length_cm"] * dim["diameter_cm"]
    else:
        real_area = None
    
    if real_area is None or real_area == 0:
        return None, None, None
    
    estimated_count = detected_area / real_area
    estimated_count = max(1, round(estimated_count))
    
    weight_per_item = dim.get("weight_g", 0)
    total_weight_g = estimated_count * weight_per_item
    total_weight_kg = total_weight_g / 1000
    
    return estimated_count, total_weight_g, total_weight_kg

def main():
    results = model("your_image_or_video.jpg")  # Change to your image or video input
    
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        names = result.names
        
        for i, box in enumerate(boxes):
            class_id = int(classes[i])
            class_name = names[class_id]
            
            qty, total_g, total_kg = estimate_quantity(box, class_name)
            if qty is not None:
                print(f"Detected {class_name}: Estimated quantity = {qty} pcs")
                print(f"Total weight = {total_g} g ({total_kg:.2f} kg)")
            else:
                print(f"Detected {class_name}: Quantity & weight estimation not available")

if __name__ == "__main__":
    main()
