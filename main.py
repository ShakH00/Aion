from machine import Pin
import time

btn = Pin(15, Pin.IN, Pin.PULL_UP)
led = Pin("LED", Pin.OUT)   # Pico W LED

while True:
    if btn.value() == 0:
        led.value(1)
        print("SCAN")
        time.sleep(0.3)
        led.value(0)
    time.sleep(0.01)
