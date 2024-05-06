ECHO "Searching for the best prompt model..."
ECHO "Testing point + box LBTD..."


ECHO "Testing points + box Camelyon..."

python evaluate_batch.py --config ../configs/vit_h/config_box.toml > ../output/vit_h/config_box.txt
