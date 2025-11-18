from flask import Flask, request, jsonify, send_from_directory
import cv2
import numpy as np
import base64
import tempfile
import os
from datetime import datetime
from scipy import ndimage

app = Flask(__name__, static_folder='static', static_url_path='')


@app.route('/')
def index():
    return app.send_static_file('index.html')


@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        if 'photo' not in request.files:
            return jsonify({'error': 'no file uploaded'}), 400

        photo = request.files['photo']
        if photo.filename == '':
            return jsonify({'error': 'no file selected'}), 400

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        tmp_path = tmp.name
        tmp.close()
        
        photo.save(tmp_path)
        result = analyze_image(tmp_path)
        
        try:
            os.unlink(tmp_path)
        except Exception as e:
            print(f"Warning: Could not delete temp file {tmp_path}: {e}")

        return jsonify(result)
    except Exception as e:
        print(f"Error in /api/analyze: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500


# ========== ANALYSIS HELPER FUNCTIONS ==========

def detect_blur(gray):
    """Detect if image is blurry using Laplacian variance"""
    try:
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        # Low variance = blurry, high variance = sharp
        if variance < 100:
            return 0.2, "Very blurry"
        elif variance < 200:
            return 0.5, "Slightly blurry"
        elif variance < 500:
            return 0.8, "Sharp"
        else:
            return 0.95, "Very sharp"
    except Exception as e:
        print(f"[DEBUG] Blur detection failed: {e}")
        return 0.7, "Unknown"


def analyze_histogram(img, hsv):
    """Analyze color distribution and histogram"""
    try:
        hist_data = {}
        
        # Calculate histograms for each channel
        for i, color in enumerate(['Blue', 'Green', 'Red']):
            hist = cv2.calcHist([img], [i], None, [256], [0, 256])
            hist_data[color] = float(np.mean(hist))
        
        # Color temperature (red vs blue dominance)
        red_avg = np.mean(img[:, :, 2])
        blue_avg = np.mean(img[:, :, 0])
        color_temp = (red_avg - blue_avg) / 255.0
        
        if color_temp > 0.15:
            temp_suggestion = "Image is too warm (reddish). Cool it down with blue/cyan light."
            temp_score = 0.6
        elif color_temp < -0.15:
            temp_suggestion = "Image is too cool (bluish). Warm it up with yellow/orange light."
            temp_score = 0.6
        else:
            temp_suggestion = "Color temperature is balanced."
            temp_score = 0.9
        
        return hist_data, color_temp, temp_suggestion, temp_score
    except Exception as e:
        print(f"[DEBUG] Histogram analysis failed: {e}")
        return {}, 0.0, "Histogram analysis unavailable.", 0.5


def detect_eyes(gray, face_rect):
    """Detect if eyes are in focus (sharp)"""
    try:
        if face_rect is None:
            return 0.7, "No face detected"
        
        x, y, w, h = face_rect
        # Eyes are roughly in upper 1/3 of face
        eye_region = gray[y:y+h//3, x:x+w]
        
        # Use Laplacian to check sharpness
        laplacian = cv2.Laplacian(eye_region, cv2.CV_64F)
        focus_score = min(1.0, laplacian.var() / 300.0)
        
        if focus_score > 0.8:
            return focus_score, "Eyes are sharp and in focus ✓"
        elif focus_score > 0.5:
            return focus_score, "Eyes are slightly soft. Try using a faster shutter speed."
        else:
            return focus_score, "Eyes are out of focus. Ensure sharp focus on eyes."
        
    except Exception as e:
        print(f"[DEBUG] Eye focus detection failed: {e}")
        return 0.7, "Eye focus analysis unavailable."


def detect_saturation(img):
    """Analyze color saturation"""
    try:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        saturation = np.mean(hsv[:, :, 1])
        
        if saturation < 30:
            return 0.5, "Colors are desaturated (muted). Consider increasing vibrance."
        elif saturation < 100:
            return 0.7, "Colors are slightly muted. Good for professional look."
        elif saturation < 180:
            return 0.9, "Colors are vibrant and balanced."
        else:
            return 0.6, "Colors are oversaturated. Reduce saturation for natural look."
    except Exception as e:
        print(f"[DEBUG] Saturation detection failed: {e}")
        return 0.7, "Saturation analysis unavailable."


def analyze_image(path):
    print(f"[DEBUG] Starting image analysis for: {path}")
    try:
        print(f"[DEBUG] Reading image...")
        img = cv2.imread(path)
        if img is None:
            print(f"[DEBUG] ERROR: Could not read image")
            return {'error': 'unable to read image'}

        h, w = img.shape[:2]
        print(f"[DEBUG] Image size: {w}x{h}")
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print(f"[DEBUG] Converted to grayscale")

        # ========== TILT DETECTION ==========
        print(f"[DEBUG] Detecting tilt...")
        angle_deg = 0.0
        try:
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
            
            if lines is not None and len(lines) > 0:
                thetas = [l[0][1] for l in lines]
                degs = [theta * 180.0 / np.pi for theta in thetas]
                degs = [((d + 90) % 180) - 90 for d in degs]
                angle_deg = float(np.median(degs))
        except Exception as e:
            print(f"[DEBUG] Tilt detection failed: {e}")
            angle_deg = 0.0

        tilt_suggestion = f'Your image is tilted by {angle_deg:.1f}°. Try adjusting the camera to level it.'
        suggested_correction = -angle_deg
        print(f"[DEBUG] Tilt angle: {angle_deg}°")

        # ========== BRIGHTNESS / LIGHTING ==========
        print(f"[DEBUG] Analyzing lighting...")
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        brightness = float(np.mean(hsv[:, :, 2]))
        
        if brightness < 60:
            lighting_suggestion = 'Underexposed: The image is too dark. Increase exposure or add more light for better visibility.'
            lighting_score = 0.4
        elif brightness < 100:
            lighting_suggestion = 'Slightly underexposed: Try adding a bit more light for better detail.'
            lighting_score = 0.6
        elif brightness > 220:
            lighting_suggestion = 'Overexposed: The image is too bright. Soften the light or reduce exposure to avoid blown highlights.'
            lighting_score = 0.4
        elif brightness > 180:
            lighting_suggestion = 'Slightly overexposed: Try reducing light intensity slightly for better balance.'
            lighting_score = 0.6
        else:
            lighting_suggestion = 'Lighting looks balanced. Well done!'
            lighting_score = 0.9

        # Analyze light distribution
        print(f"[DEBUG] Analyzing light distribution...")
        try:
            left_half = np.mean(hsv[:, :w//2, 2])
            right_half = np.mean(hsv[:, w//2:, 2])
            light_balance = abs(left_half - right_half)
            
            if light_balance > 40:
                if left_half > right_half:
                    lighting_suggestion += ' The light source is too harsh on the left side. Move the light to the right for better balance.'
                else:
                    lighting_suggestion += ' The light source is too harsh on the right side. Move the light to the left for better balance.'
        except Exception as e:
            print(f"[DEBUG] Light distribution analysis failed: {e}")

        print(f"[DEBUG] Brightness: {brightness}, Lighting score: {lighting_score}")

        # ========== FACE DETECTION & POSE ==========
        print(f"[DEBUG] Detecting faces...")
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
        
        pose_suggestion = None
        face_info = None
        scene_type = 'landscape'
        
        if len(faces) > 0:
            print(f"[DEBUG] Found {len(faces)} faces")
            scene_type = 'portrait'
            x, y, fw, fh = sorted(faces, key=lambda r: r[2] * r[3], reverse=True)[0]
            cx, cy = int(x + fw / 2), int(y + fh / 2)
            face_info = {'x': int(x), 'y': int(y), 'w': int(fw), 'h': int(fh), 'cx': cx, 'cy': cy}
            
            face_area_ratio = (fw * fh) / (w * h)
            
            if cx < w * 0.3:
                position_msg = 'left third'
                emotion_tip = 'This angle looks powerful. For a more confident shot, try moving the subject slightly right.'
            elif cx > w * 0.7:
                position_msg = 'right third'
                emotion_tip = 'This angle looks confident. For an intimate shot, try centering the subject more.'
            else:
                position_msg = 'center'
                emotion_tip = 'Subject is well-centered. Consider moving left or right for a more dynamic composition.'

            if face_area_ratio < 0.1:
                pose_suggestion = f"Subject is {position_msg} and quite small in frame. Move closer or use a higher angle to improve the composition. {emotion_tip}"
            elif face_area_ratio > 0.4:
                pose_suggestion = f"Subject fills most of the frame. Try moving back slightly to add context. {emotion_tip}"
            else:
                pose_suggestion = f"Subject's posture and position look good. {emotion_tip}"
        else:
            print(f"[DEBUG] No faces detected")
            pose_suggestion = 'No clear face detected. For portrait photography, try to center your subject and ensure good lighting on the face.'
            scene_type = 'landscape'

        print(f"[DEBUG] Scene type: {scene_type}")

        # ========== BLUR DETECTION ==========
        print(f"[DEBUG] Detecting blur...")
        blur_score, blur_msg = detect_blur(gray)

        # ========== COLOR ANALYSIS ==========
        print(f"[DEBUG] Analyzing color...")
        hist_data, color_temp, temp_suggestion, temp_score = analyze_histogram(img, hsv)
        
        # ========== SATURATION ANALYSIS ==========
        print(f"[DEBUG] Analyzing saturation...")
        sat_score, sat_msg = detect_saturation(img)

        # ========== EYE FOCUS (Portrait-specific) ==========
        eye_focus_score = 0.7
        eye_focus_msg = "N/A (not a portrait)"
        if scene_type == 'portrait' and face_info:
            print(f"[DEBUG] Detecting eye focus...")
            eye_focus_score, eye_focus_msg = detect_eyes(gray, (face_info['x'], face_info['y'], face_info['w'], face_info['h']))
        print(f"[DEBUG] Calculating composition score...")
        composition_score = 0.7
        if face_info:
            fx, fy = face_info['cx'], face_info['cy']
            third_x = [w / 3, 2 * w / 3]
            third_y = [h / 3, 2 * h / 3]
            try:
                min_dist = min([
                    np.sqrt((fx - tx)**2 + (fy - ty)**2)
                    for tx in third_x for ty in third_y
                ])
                max_dist = np.sqrt(w**2 + h**2)
                composition_score = 0.9 - (0.2 * min_dist / max_dist)
            except Exception as e:
                print(f"[DEBUG] Composition scoring failed: {e}")

        print(f"[DEBUG] Composition score: {composition_score}")

        # ========== BUILD OVERLAY ==========
        print(f"[DEBUG] Building overlay image...")
        overlay = img.copy()
        
        # Rule of thirds grid
        color_grid = (0, 255, 0)
        thickness = 2
        for i in (1, 2):
            cv2.line(overlay, (int(w * i / 3), 0), (int(w * i / 3), h), color_grid, thickness)
            cv2.line(overlay, (0, int(h * i / 3)), (w, int(h * i / 3)), color_grid, thickness)

        # Tilt annotation
        cv2.putText(overlay, f'Tilt: {angle_deg:.1f}°', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(overlay, f'Scene: {scene_type.capitalize()}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 200), 1)

        # Face annotation
        if face_info:
            fx, fy, fw, fh = face_info['x'], face_info['y'], face_info['w'], face_info['h']
            cv2.rectangle(overlay, (fx, fy), (fx + fw, fy + fh), (255, 0, 0), 2)
            
            # Suggested target with direction labels
            tx = int(w / 3) if face_info['cx'] < w / 2 else int(w * 2 / 3)
            ty = int(h / 3)
            
            # Determine direction
            if face_info['cx'] < w / 3:
                direction = "Move RIGHT"
            elif face_info['cx'] > 2 * w / 3:
                direction = "Move LEFT"
            else:
                direction = "Move UP"
            
            # Draw target circle
            cv2.circle(overlay, (tx, ty), 12, (0, 0, 255), -1)
            cv2.arrowedLine(overlay, (face_info['cx'], face_info['cy']), (tx, ty), (0, 0, 255), 2, tipLength=0.05)
            
            # Add direction text label
            cv2.putText(overlay, direction, (int(w/2 - 80), int(h - 40)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.putText(overlay, "to Rule of Thirds", (int(w/2 - 120), int(h - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        print(f"[DEBUG] Encoding overlay...")
        # Encode overlay as base64
        _, buf = cv2.imencode('.png', overlay)
        overlay_b64 = base64.b64encode(buf).decode('ascii')
        overlay_data_url = 'data:image/png;base64,' + overlay_b64

        # ========== BUILD SUGGESTIONS ==========
        print(f"[DEBUG] Building suggestions...")
        suggestions = []
        suggestions.append({'key': 'tilt', 'message': tilt_suggestion, 'category': 'Angle', 'icon': '📐'})
        suggestions.append({'key': 'lighting', 'message': lighting_suggestion, 'category': 'Lighting', 'icon': '💡'})
        suggestions.append({'key': 'pose', 'message': pose_suggestion, 'category': 'Composition', 'icon': '📍'})
        suggestions.append({'key': 'blur', 'message': blur_msg, 'category': 'Focus', 'icon': '🔍', 'score': blur_score})
        suggestions.append({'key': 'color_temp', 'message': temp_suggestion, 'category': 'Color', 'icon': '🌡️', 'score': temp_score})
        suggestions.append({'key': 'saturation', 'message': sat_msg, 'category': 'Vibrancy', 'icon': '✨', 'score': sat_score})
        
        if scene_type == 'portrait':
            suggestions.append({'key': 'eye_focus', 'message': eye_focus_msg, 'category': 'Portrait', 'icon': '👁️', 'score': eye_focus_score})
        
        print(f"[DEBUG] Returning result...")
        result = {
            'tilt_degrees': float(angle_deg),
            'suggested_correction': float(suggested_correction),
            'brightness': float(brightness),
            'blur_score': float(blur_score),
            'saturation_score': float(sat_score),
            'eye_focus_score': float(eye_focus_score) if scene_type == 'portrait' else 0.0,
            'lighting_score': float(lighting_score),
            'composition_score': float(composition_score),
            'scene_type': scene_type,
            'suggestions': suggestions,
            'overlay_image': overlay_data_url,
            'face': face_info,
        }
        print(f"[DEBUG] Analysis complete!")
        return result
        
    except Exception as e:
        print(f"[DEBUG] EXCEPTION in analyze_image: {e}")
        import traceback
        traceback.print_exc()
        return {'error': f'Image analysis failed: {str(e)}'}


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
