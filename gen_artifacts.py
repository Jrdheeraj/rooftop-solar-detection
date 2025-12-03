import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
import json

"""
Sample Artifact Generator for EcoInnovators Submission
Creates 2 example overlay visualizations showing:
  - Input rooftop image
  - Bounding boxes around detected panels
  - Confidence scores
  - Estimated panel area
  - QC status
"""

def create_sample_overlay_verifiable():
    """
    Create a sample overlay for a rooftop WHERE SOLAR PANELS ARE DETECTED (VERIFIABLE)
    """
    
    # Create a synthetic rooftop image (400x400)
    # In reality, this would be a satellite image
    img = np.ones((400, 400, 3), dtype=np.uint8) * 240  # Light gray background
    
    # Add some "roof texture" (noise)
    noise = np.random.randint(0, 20, (400, 400, 3), dtype=np.uint8)
    img = cv2.add(img, noise)
    
    # Draw some building structures
    cv2.rectangle(img, (50, 50), (350, 350), (200, 180, 160), -1)  # Roof color
    
    # Draw simulated solar panels (dark blue rectangles)
    # Panel 1: Top-left area
    cv2.rectangle(img, (70, 80), (180, 150), (50, 100, 200), -1)   # Dark blue panel
    
    # Panel 2: Top-right area
    cv2.rectangle(img, (220, 90), (320, 160), (50, 100, 200), -1)
    
    # Panel 3: Bottom area
    cv2.rectangle(img, (100, 200), (300, 280), (50, 100, 200), -1)
    
    # Add grid lines on panels to make them look realistic
    for x in range(70, 180, 30):
        cv2.line(img, (x, 80), (x, 150), (30, 80, 180), 1)
    for y in range(80, 150, 25):
        cv2.line(img, (70, y), (180, y), (30, 80, 180), 1)
    
    # Create matplotlib figure for overlay
    fig, ax = plt.subplots(figsize=(12, 12), dpi=100)
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.set_title('🌞 Solar Panel Detection - VERIFIABLE (Panels Detected)', 
                 fontsize=16, fontweight='bold', color='green')
    
    # Draw bounding boxes for detected panels
    # Panel 1 bbox
    rect1 = Rectangle((70, 80), 110, 70, linewidth=3, edgecolor='lime', 
                      facecolor='none', label='Detected Panel')
    ax.add_patch(rect1)
    ax.text(75, 70, 'Confidence: 0.94', fontsize=11, color='lime', 
            fontweight='bold', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    ax.text(75, 160, 'Area: 15.2 m²', fontsize=11, color='lime', 
            fontweight='bold', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    # Panel 2 bbox
    rect2 = Rectangle((220, 90), 100, 70, linewidth=3, edgecolor='lime', 
                      facecolor='none')
    ax.add_patch(rect2)
    ax.text(225, 80, 'Confidence: 0.91', fontsize=11, color='lime', 
            fontweight='bold', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    ax.text(225, 160, 'Area: 14.5 m²', fontsize=11, color='lime', 
            fontweight='bold', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    # Panel 3 bbox
    rect3 = Rectangle((100, 200), 200, 80, linewidth=3, edgecolor='lime', 
                      facecolor='none')
    ax.add_patch(rect3)
    ax.text(105, 190, 'Confidence: 0.88', fontsize=11, color='lime', 
            fontweight='bold', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    ax.text(105, 290, 'Area: 32.1 m²', fontsize=11, color='lime', 
            fontweight='bold', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    # Add metadata box
    metadata_text = (
        'Sample ID: 1067\n'
        'Location: 12.9716°N, 77.5946°E\n'
        'Has Solar: TRUE ✓\n'
        'Total Panel Area: 61.8 m²\n'
        'Buffer Radius: 1200 sq.ft (6m)\n'
        'QC Status: VERIFIABLE ✓\n'
        'Image Source: Synthetic'
    )
    ax.text(10, 360, metadata_text, fontsize=10, color='white', 
            fontweight='bold', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='darkgreen', alpha=0.85, 
                     edgecolor='lime', linewidth=2))
    
    # Legend
    ax.text(10, 30, '🟩 = Detected Solar Panel\n✓ = Verified Detection\nGreen = High Confidence', 
            fontsize=10, color='white', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='darkblue', alpha=0.85, 
                     edgecolor='cyan', linewidth=2))
    
    ax.axis('off')
    plt.tight_layout()
    
    # Save figure
    output_path = Path('outputs/sample_overlay_verifiable.jpg')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()
    
    return str(output_path)


def create_sample_overlay_not_verifiable():
    """
    Create a sample overlay for a rooftop WHERE NO SOLAR PANELS ARE DETECTED (NOT_VERIFIABLE)
    """
    
    # Create a synthetic rooftop image
    img = np.ones((400, 400, 3), dtype=np.uint8) * 235
    
    # Add noise
    noise = np.random.randint(0, 15, (400, 400, 3), dtype=np.uint8)
    img = cv2.add(img, noise)
    
    # Draw roof base
    cv2.rectangle(img, (40, 40), (360, 360), (190, 170, 150), -1)  # Roof color
    
    # Add obstacles (trees, water tank, etc.)
    cv2.circle(img, (150, 120), 50, (100, 180, 70), -1)   # Green circle (tree/vegetation)
    cv2.rectangle(img, (250, 100), (320, 180), (100, 120, 130), -1)  # Gray (water tank)
    
    # Add heavy clouds/shadow effect
    for i in range(0, 400, 50):
        cv2.line(img, (i, 200), (i+40, 250), (180, 180, 180), 15)
        cv2.line(img, (i+100, 280), (i+140, 330), (180, 180, 180), 15)
    
    # Create matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 12), dpi=100)
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.set_title('🌞 Solar Panel Detection - NOT VERIFIABLE (Ambiguous/Obstructed)', 
                 fontsize=16, fontweight='bold', color='orange')
    
    # Mark obstacles
    circle1 = plt.Circle((150, 120), 50, color='red', fill=False, linewidth=3, 
                        linestyle='--', label='Obstruction')
    ax.add_patch(circle1)
    ax.text(120, 50, 'Tree/Vegetation', fontsize=11, color='red', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    rect_obstacle = Rectangle((250, 100), 70, 80, linewidth=3, edgecolor='red', 
                              facecolor='none', linestyle='--')
    ax.add_patch(rect_obstacle)
    ax.text(255, 85, 'Water Tank', fontsize=11, color='red', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    # Add metadata box
    metadata_text = (
        'Sample ID: 2045\n'
        'Location: 13.0827°N, 77.5934°E\n'
        'Has Solar: FALSE\n'
        'Panel Area: 0.0 m²\n'
        'Buffer Radius: 1200 sq.ft (6m)\n'
        'QC Status: NOT_VERIFIABLE ⚠️\n'
        'Reason: Heavy shadow/cloud cover\n'
        'Image Source: Synthetic'
    )
    ax.text(10, 360, metadata_text, fontsize=10, color='white', fontweight='bold',
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='darkorange', alpha=0.85, 
                     edgecolor='yellow', linewidth=2))
    
    # Legend
    ax.text(10, 30, '🔴 = Obstruction/Ambiguity\n⚠️ = Cannot Verify\nOrange = Insufficient Evidence', 
            fontsize=10, color='white', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='darkred', alpha=0.85, 
                     edgecolor='yellow', linewidth=2))
    
    ax.axis('off')
    plt.tight_layout()
    
    # Save figure
    output_path = Path('outputs/sample_overlay_not_verifiable.jpg')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()
    
    return str(output_path)


def create_sample_predictions_json():
    """
    Create a sample predictions JSON file following the challenge schema
    """
    
    sample_predictions = [
        {
            "sample_id": 1067,
            "lat": 12.9716,
            "lon": 77.5946,
            "has_solar": True,
            "confidence": 0.92,
            "pv_area_sqm_est": 61.8,
            "buffer_radius_sqft": 1200,
            "qc_status": "VERIFIABLE",
            "bbox_or_mask": "POLYGON ((70 80, 180 80, 180 150, 70 150, 70 80))",
            "image_metadata": {
                "source": "synthetic_static",
                "capture_date": "2024-12-03"
            }
        },
        {
            "sample_id": 2045,
            "lat": 13.0827,
            "lon": 77.5934,
            "has_solar": False,
            "confidence": 0.0,
            "pv_area_sqm_est": 0.0,
            "buffer_radius_sqft": 1200,
            "qc_status": "NOT_VERIFIABLE",
            "bbox_or_mask": "[]",
            "image_metadata": {
                "source": "synthetic_static",
                "capture_date": "2024-12-03",
                "reason": "Heavy shadow/cloud cover; tree obstruction visible"
            }
        },
        {
            "sample_id": 1523,
            "lat": 12.8765,
            "lon": 77.6234,
            "has_solar": True,
            "confidence": 0.87,
            "pv_area_sqm_est": 42.5,
            "buffer_radius_sqft": 1200,
            "qc_status": "VERIFIABLE",
            "bbox_or_mask": "POLYGON ((100 100, 250 100, 250 200, 100 200, 100 100))",
            "image_metadata": {
                "source": "synthetic_static",
                "capture_date": "2024-12-03"
            }
        }
    ]
    
    output_path = Path('outputs/sample_predictions.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(sample_predictions, f, indent=2)
    
    print(f"✅ Saved: {output_path}")
    return str(output_path)


if __name__ == '__main__':
    print("📝 Generating sample artifacts for EcoInnovators submission...\n")
    
    # Create overlays
    overlay1 = create_sample_overlay_verifiable()
    print()
    overlay2 = create_sample_overlay_not_verifiable()
    print()
    
    # Create sample JSON
    json_file = create_sample_predictions_json()
    print()
    
    print("=" * 60)
    print("✅ All sample artifacts created successfully!")
    print("=" * 60)
    print("\n📁 Files created:")
    print(f"  1. {overlay1}")
    print(f"  2. {overlay2}")
    print(f"  3. {json_file}")
    print("\n🚀 Ready for submission!")
