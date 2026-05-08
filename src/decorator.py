import time

# using manual decorator
# def brew_tea():
#     print("Brewing tea...")
#     time.sleep(1)
#     print("Tea is ready!")

def timer_dec2(base_fn):
    start_time = time.time()
    base_fn()
    end_time = time.time()
    print(f"Task time: {end_time - start_time} seconds")

def timer_dec(base_fn):
    def enhanced_fn():
        start_time = time.time()
        base_fn()
        end_time = time.time()
        print(f"Task time: {end_time - start_time} seconds")
    return enhanced_fn

# brew_tea()
# timer_dec2(brew_tea)
# brew_tea = timer_dec(brew_tea)
# brew_tea()

@timer_dec
def brew_tea():
    print("Brewing tea...")
    time.sleep(1)
    print("Tea is ready!")

brew_tea()
