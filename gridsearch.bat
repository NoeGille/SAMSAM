ECHO "Searching for the best prompt model..."
ECHO "Testing point + box LBTD..."

python evaluate_batch.py --config ../configs/prompt/centered/box_point1/configLBTDMEDSAM.toml > ../output/prompt/centered/box_point1/configLBTDMEDSAM.txt
python evaluate_batch.py --config ../configs/prompt/centered/box_point1/configLBTDSAM.toml > ../output/prompt/centered/box_point1/configLBTDSAM.txt
python evaluate_batch.py --config ../configs/prompt/centered/box_point5/configLBTDMEDSAM.toml > ../output/prompt/centered/box_point5/configLBTDMEDSAM.txt
python evaluate_batch.py --config ../configs/prompt/centered/box_point5/configLBTDSAM.toml > ../output/prompt/centered/box_point5/configLBTDSAM.txt

ECHO "Testing point LBTD..."

python evaluate_batch.py --config ../configs/prompt/centered/point1/configLBTDMEDSAM.toml > ../output/prompt/centered/point1/configLBTDMEDSAM.txt
python evaluate_batch.py --config ../configs/prompt/centered/point1/configLBTDSAM.toml > ../output/prompt/centered/point1/configLBTDSAM.txt
python evaluate_batch.py --config ../configs/prompt/centered/point5/configLBTDMEDSAM.toml > ../output/prompt/centered/point5/configLBTDMEDSAM.txt
python evaluate_batch.py --config ../configs/prompt/centered/point5/configLBTDSAM.toml > ../output/prompt/centered/point5/configLBTDSAM.txt

ECHO "Testing points + box Camelyon..."

python evaluate_batch.py --config ../configs/prompt/centered/box_point1/configcamelyonMEDSAM.toml > ../output/prompt/centered/box_point1/configcamelyonMEDSAM.txt
python evaluate_batch.py --config ../configs/prompt/centered/box_point1/configcamelyonSAM.toml > ../output/prompt/centered/box_point1/configcamelyonSAM.txt
python evaluate_batch.py --config ../configs/prompt/centered/box_point5/configcamelyonMEDSAM.toml > ../output/prompt/centered/box_point5/configcamelyonMEDSAM.txt
python evaluate_batch.py --config ../configs/prompt/centered/box_point5/configcamelyonSAM.toml > ../output/prompt/centered/box_point5/configcamelyonSAM.txt

ECHO "Testing points Camelyon..."

python evaluate_batch.py --config ../configs/prompt/centered/point1/configcamelyonMEDSAM.toml > ../output/prompt/centered/point1/configcamelyonMEDSAM.txt
python evaluate_batch.py --config ../configs/prompt/centered/point1/configcamelyonSAM.toml > ../output/prompt/centered/point1/configcamelyonSAM.txt
python evaluate_batch.py --config ../configs/prompt/centered/point5/configcamelyonMEDSAM.toml > ../output/prompt/centered/point5/configcamelyonMEDSAM.txt
python evaluate_batch.py --config ../configs/prompt/centered/point5/configcamelyonSAM.toml > ../output/prompt/centered/point5/configcamelyonSAM.txt
