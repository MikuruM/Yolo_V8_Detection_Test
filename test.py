import pyautogui

screen_width, screen_height = pyautogui.size()
monitor = {
    "top": 0,
    "left": 0,
    "width": screen_width,
    "height": screen_height
}

print(monitor)