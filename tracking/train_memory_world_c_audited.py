"""Stage C entry point using the existing accelerated full manifest audit."""
import train_memory_world as trainer
from world_manifest_fast import read_manifest

trainer.read_manifest = read_manifest

if __name__ == '__main__':
    trainer.main()
