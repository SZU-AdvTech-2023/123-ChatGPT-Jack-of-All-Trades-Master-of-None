import os
import sys

project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# wsd_evaluation_framework = '/home/liyuncong/project/wsd/data/WSD_Evaluation_Framework'
wsd_evaluation_framework = os.path.join(project_dir, 'data', 'WSD_Evaluation_Framework')
# scorer_path = os.path.join(project_dir, 'datasets/WSD_Evaluation_Framework/Evaluation_Datasets/Scorer.java')
scorer_path = os.path.join(project_dir, 'datasets\\WSD_Evaluation_Framework\\Evaluation_Datasets\\Scorer.java')

if __name__ == '__main__':
    print(project_dir)
    print(wsd_evaluation_framework)
    print(scorer_path)
