#!/bin/bash
set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}Lab 6: Model Training Pipeline${NC}"
echo "======================================"

if [ ! -f "models/rating_features.pkl" ]; then
    echo -e "${RED}Error: models/rating_features.pkl not found${NC}"
    echo "Please run Lab 5 first"
    exit 1
fi

if [ ! -f "data/processed/ratings_clean.csv" ]; then
    echo -e "${RED}Error: data/processed/ratings_clean.csv not found${NC}"
    exit 1
fi

if [ -z "$VIRTUAL_ENV" ]; then
    source .venv/bin/activate
fi

echo -e "${BLUE}Running training pipeline with hyperparameter tuning...${NC}"

python src/train_main.py \
    --features_path models/rating_features.pkl \
    --ratings_path data/processed/ratings_clean.csv \
    --model_dir models \
    --tune \
    --k_values 3 5 10 15 20

if [ -f "models/model.pkl" ] && [ -f "models/metadata.json" ]; then
    echo -e "${GREEN}✓ Model and metadata created${NC}"
    echo ""
    echo "Model metadata:"
    cat models/metadata.json | python -m json.tool 2>/dev/null || cat models/metadata.json
else
    echo -e "${RED}Error: Model files not created${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Training pipeline complete!${NC}"