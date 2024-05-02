ECHO "Searching for the best prompt model..."
ECHO "Testing 1 point..."

python evaluate_batch.py --config ../configs/prompt/point1/configLBTDMEDSAM.toml > ../output/prompt/point1/configLBTDMEDSAM.txt
python evaluate_batch.py --config ../configs/prompt/point1/configLBTDSAM.toml > ../output/prompt/point1/configLBTDSAM.txt

python evaluate_batch.py --config ../configs/prompt/point1/configcamelyonMEDSAM.toml > ../output/prompt/point1/configcamelyonMEDSAM.txt
python evaluate_batch.py --config ../configs/prompt/point1/configcamelyonSAM.toml > ../output/prompt/point1/configcamelyonSAM.txt

ECHO "Testing 5 points..."

python evaluate_batch.py --config ../configs/prompt/point5/configLBTDMEDSAM.toml > ../output/prompt/point5/configLBTDMEDSAM.txt
python evaluate_batch.py --config ../configs/prompt/point5/configLBTDSAM.toml > ../output/prompt/point5/configLBTDSAM.txt

python evaluate_batch.py --config ../configs/prompt/point5/configcamelyonMEDSAM.toml > ../output/prompt/point5/configcamelyonMEDSAM.txt
python evaluate_batch.py --config ../configs/prompt/point5/configcamelyonSAM.toml > ../output/prompt/point5/configcamelyonSAM.txt