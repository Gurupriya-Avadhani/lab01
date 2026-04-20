# #!/bin/bash
# set -e

# GREEN='\033[0;32m'
# BLUE='\033[0;34m'
# RED='\033[0;31m'
# NC='\033[0m'

# echo -e "${BLUE}Lab 7: Model Evaluation Pipeline${NC}"
# echo "===================================="

# if [ ! -f "models/model.pkl" ]; then
#     echo -e "${RED}Error: models/model.pkl not found${NC}"
#     exit 1
# fi

# if [ ! -f "models/metadata.json" ]; then
#     echo -e "${RED}Error: models/metadata.json not found${NC}"
#     exit 1
# fi

# if [ ! -f "data/processed/ratings_clean.csv" ]; then
#     echo -e "${RED}Error: data/processed/ratings_clean.csv not found${NC}"
#     exit 1
# fi

# if [ -z "$VIRTUAL_ENV" ] && [ -f ".venv/bin/activate" ]; then
#     source .venv/bin/activate
# fi

# echo -e "${BLUE}Running evaluation pipeline...${NC}"

# python src/evaluate_main.py \
#     --model_path models/model.pkl \
#     --metadata_path models/metadata.json \
#     --test_path data/processed/ratings_clean.csv \
#     --ratings_path data/processed/ratings_clean.csv \
#     --n_movies 100 \
#     --eval_dir evaluations

# REPORT="evaluations/evaluation_report.json"

# if [ -f "$REPORT" ]; then
#     echo -e "${GREEN}✓ Evaluation report created${NC}"
#     echo ""

#     python - <<EOF 2>/dev/null
# import json
# with open("$REPORT") as f:
#     r = json.load(f)

# m = r.get("rating_prediction", {})
# c = r.get("coverage", {})
# b = r.get("baselines", {})

# print("Report summary:")
# print(f"  RMSE:              {m.get('rmse', 0):.4f}")
# print(f"  MAE:               {m.get('mae', 0):.4f}")
# print(f"  Catalog Coverage:  {c.get('coverage_ratio', 0):.2%}")
# print(f"  Best Baseline:     {b.get('best_baseline', 0):.4f}")
# EOF

# else
#     echo -e "${RED}Error: Evaluation report not created${NC}"
#     exit 1
# fi

# echo -e "${GREEN}✓ Evaluation complete!${NC}"



#!/bin/bash
set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}Lab 7: Model Evaluation Pipeline${NC}"
echo "===================================="

if [ ! -f "models/model.pkl" ]; then
    echo -e "${RED}Error: models/model.pkl not found${NC}"
    exit 1
fi

if [ ! -f "models/metadata.json" ]; then
    echo -e "${RED}Error: models/metadata.json not found${NC}"
    exit 1
fi

if [ ! -f "data/processed/ratings_clean.csv" ]; then
    echo -e "${RED}Error: data/processed/ratings_clean.csv not found${NC}"
    exit 1
fi

if [ -z "$VIRTUAL_ENV" ] && [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

echo -e "${BLUE}Running evaluation pipeline...${NC}"

PYTHONPATH=$(pwd) python src/evaluate_main.py \
    --model_path models/model.pkl \
    --metadata_path models/metadata.json \
    --test_path data/processed/ratings_clean.csv \
    --ratings_path data/processed/ratings_clean.csv \
    --n_movies 100 \
    --eval_dir evaluations

REPORT="evaluations/evaluation_report.json"

if [ -f "$REPORT" ]; then
    echo -e "${GREEN}✓ Evaluation report created${NC}"
    echo ""

    python - <<EOF 2>/dev/null
import json
with open("$REPORT") as f:
    r = json.load(f)

m = r.get("rating_prediction", {})
c = r.get("coverage", {})
b = r.get("baselines", {})

print("Report summary:")
print(f"  RMSE:              {m.get('rmse', 0):.4f}")
print(f"  MAE:               {m.get('mae', 0):.4f}")
print(f"  Catalog Coverage:  {c.get('coverage_ratio', 0):.2%}")
print(f"  Best Baseline:     {b.get('best_baseline', 0):.4f}")
EOF

else
    echo -e "${RED}Error: Evaluation report not created${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Evaluation complete!${NC}"