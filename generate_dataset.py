import random
import math

def generate_dataset(filename="dataset.csv", n_samples=2000):
    with open(filename, "w") as f:
        f.write("x,y,label\n")
        
        for _ in range(n_samples):
            x = random.uniform(-1.0, 1.0)
            y = random.uniform(-1.0, 1.0)
            
            # Calculate distance from center
            distance = math.sqrt(x**2 + y**2)
            
            # Label: 1.0 if inside circle (radius 0.7), -1.0 if outside
            # We add a tiny bit of "noise" to the boundary to make it realistic
            # but for a first test, a clean boundary is better.
            if distance < 0.7:
                label = 1.0
            else:
                label = -1.0
                
            f.write(f"{x:.4f},{y:.4f},{label:.1f}\n")

if __name__ == "__main__":
    generate_dataset()
    print("Generated dataset.csv with 2000 samples.")
