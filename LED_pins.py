import RPi.GPIO as io
import time


# pins on raspberry pi
pins = [22, 23, 24, 27]


def set_all_low() -> None:
    for pin in pins:
        print(pin)
        io.setup(pin, io.OUT)
        io.output(pin, io.LOW)


def set_all_high() -> None:
    for pin in pins:
        io.setup(pin, io.OUT)
        io.output(pin, io.HIGH)

if __name__ == "__main__":
    io.setmode(io.BCM)
    io.setwarnings(False)
    #set_all_high()
    #time.sleep(10)
    set_all_low()
