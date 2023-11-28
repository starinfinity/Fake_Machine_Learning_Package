import threading
import time

# Creating a global pool
global processors

def progression_bar(total_time=20):
    num_bar = 50
    sleep_intvl = total_time/num_bar
    print("")
    for i in range(1,num_bar):
        print("\r", end="")
        print("Training: {:.1%} ".format(i/num_bar), end="")
        time.sleep(sleep_intvl)

def commencing_model():
    # Create a thread
    waiting_thread = threading.Thread(target=_simulate_progress)
    waiting_thread.start()
    global processors
    print("-" * 50)
    print("Creating threads for parallel load!")
    print("-" * 50)
    print("Updating global pool with processor count..")

    # Wait for the waiting thread to finish before proceeding
    waiting_thread.join()

    print("Releasing threads to global pool!")

def _simulate_progress():
    # Simulating non-blocking operations
    time.sleep(1)
    progression_bar()
    time.sleep(1)
    print("\n")

