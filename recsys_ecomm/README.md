![cover_photo](reports/figures/recsys_cover.png)


# AgentRex: Multi-Strategy E-Commerce Recommendation Engine

*In the dynamic landscape of e-commerce and retail, recommendation systems are integral to enhancing customer experiences and fostering business growth.*

*As consumers navigate millions of online products, they often encounter information overload, which complicates the process of identifying items that align with their interests. The primary challenge is effectively filtering this abundance of information while delivering personalized recommendations that are relevant, engaging, and tailored to individual preferences. The ability to quickly provide relevant and personalized suggestions is not only a technical challenge but also a critical competitive advantage.*

*Recommendation engines address this need by generating personalized product suggestions based on user behavior, preferences, item attributes, and various other factors. These systems are extensively utilized in the retail sector to enhance customer engagement, elevate average order values, and improve conversion rates. Prominent companies, such as Amazon and Walmart, leverage sophisticated recommendation algorithms to optimize inventory turnover and maximize revenue per user.*

*This capstone project focuses on the development and evaluation of a multi-strategy recommendation engine utilizing real-world data. It encompasses the complete process—from data preprocessing and exploratory data analysis to model selection, evaluation, and optimization—highlighting the practical application of machine learning techniques in the creation of advanced recommendation engines.*

---
## 1. Data

**Objectives:**

The project focuses on creating a multi-strategy recommendation engine that leverages machine learning algorithms to provide personalized recommendations. The main objectives are to:

- Improve the accuracy of predicting user preferences by using advanced algorithms such as collaborative filtering, content-based filtering, or hybrid approaches.

- Develop methods to address the cold-start problem for both new users and new items.


**Scope of Solution Space:**


<!-- Similar to a hybrid recommender system multi-strategy combines multiple recommendation techniques to improve accuracy, overcome limitations of individual methods, and enhance user experience. 

A multi-strategy recommender engine:*
- Leverages multiple algorithms (CF, CBF, popularity, etc.)
- Increases recommendation robustness and personalization
- Helps overcome cold start, sparsity, and overfitting
- Can be tuned dynamically based on use case and user feedback

Unlike the hybrid recommenders which are tightly coupled models and methods, A multi-strategy recommender emphasizes loosely-coupled, independent models where different the algorithms operate independently, with the methods being modular, and with their results that can be strategically combined -->

The proof of concept (POC) will be developed using Jupyter Notebooks until we achieve optimal model performance.

It's important to clarify that this project does not include the development of any front-end components for an eCommerce website. The next step, which involves integrating the model into an eCommerce platform, should be planned as a separate project.

**Data Source:** [UCSD Amazon Reviews 2023 | McAuley Labs](https://amazon-reviews-2023.github.io/)

The dataset includes a large collection of user reviews, product metatdata, and product links collected in 2023 by [McAuley Lab](https://cseweb.ucsd.edu/~jmcauley/)\
Detailed description of the fields can be found in the aforementioned link.\
Due to technical resource constraints, only the appliances data were extracted and further resampled for this analysis. The notebook extracts the appliances json files from the following URLs:

[Amazon Users Reviews](https://mcauleylab.ucsd.edu:8443/public_datasets/data/amazon_2023/raw/review_categories/Appliances.jsonl.gz)\
[Appliances Metadata](https://mcauleylab.ucsd.edu:8443/public_datasets/data/amazon_2023/raw/meta_categories/meta_Appliances.jsonl.gz)

A user-defined function reads the json files in manageable chunks, samples and combines them. There is also an option to read json files uploaded locally. 

## 2. Data Cleaning

As the focus of the project is in recommendation, only the ratings, user-related identifiers, product attributes and metadata info were retained; everything else were dropped. 
There is a decision block which requires setting a threshold on the % of empty rows that can be used as a criteria for dropping. Initially automated to drop columns that exceed the selected threshold but resorted to manual. 

- **Meta Data**:
  - Dropped unnecessary columns for NLP and  (e.g., images, videos, price, average_rating, etc.).
  - Checked and filled missing values in key columns (`store`, `main_category`).
  - Extracted `subcategory` from the last element of the `categories` list or used `main_category` as fallback.
  - Ensured no nulls in critical columns after cleaning.

- **Ratings Data**:
  - Dropped columns not needed for modeling (e.g., title, text, images, timestamp, etc.).
  - Verified no nulls in essential columns.

- **Merged** product metadata and ratings on `parent_asin`.  





## 3. Exploratory Data Analysis (EDA)

- **Summary Statistics**:
  - Unique products: 27–30 (depending on cleaning stage)
  - Unique users: ~5363
  - Total ratings: ~5567
  - Average ratings per product: ~206.93
  - Average ratings per user: ~1.04

- **Category Insights**:
  - For this specific run, most ratings are concentrated in a few main categories (e.g., "Amazon Home", "Appliances", "Grocery")

    ![main_cats](reports/figures/maincats_barh.png)

  - Subcategories like "Reusable Filters" and "Permanent Filters" are prominent.
    ![sub_cats](reports/figures/subcats_barh.png)

- **Ratings Distributions**:

  - Plotted rating distributions and counts.

    ![rats_kde](reports/figures/ratings_kdeplot.png)
    ![rats_count](reports/figures/ratings_countplot.png)

  - Ratings by ASIN and User ID.

    ![rats_asin](reports/figures/ratings_by_asin.png)
    ![rats_user](reports/figures/ratings_by_user_id.png)

- **NLP Insights** 

  - Distribution of word lengths in product titles, descriptions, and features.

    ![wc_title](reports/figures/word_count_title.png)

    ![wc_feats](reports/figures/word_count_features.png)

    ![wc_desc](reports/figures/word_count_description.png)

    > Note: The outliers were included in the NLP processing for this specific run. Should there be technical constraints, the decision to process the outliers should be aligned with the stakeholders and technical owners. 


## 4. Feature Engineering

- **Feature Group**: Combined `parent_asin`, `main_category`, `subcategory`, `maker`, and `title` into a single string for NLP-based similarity.
- **Text Processing**: Applied preprocessing, tokenization, and lemmatization to the feature group.

## 5. Content-Based Filtering

- **TF-IDF Vectorization**: Vectorized the processed feature group.
- **Cosine Similarity**: Calculated pairwise similarities between products.


### Option A: Similar to ASIN ID (key)
Find similar items by ASIN
- Sample ASIN: ***B000DLB2FI***

**Top 10 Similar to Item:**

|parent_asin|main_category           |subcategory       |maker         |title                                                                                                                                                                                                   |similarity         |
|-----------|------------------------|------------------|--------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------|
|B07P3Y8JWY |Amazon Home             |Reusable Filters  |Simple Cups   |Disposable Paper Coffee Filters 600 count - Compatible with Keurig, K-Cup machines & other Single Serve Coffee Brewer Reusable K Cups - Use Your Own Coffee & Make Your Own Pods - Works with All Brands|0.4219590337271575 |
|B07RNJY499 |Amazon Home             |Reusable Filters  |iPartsPlusMore|iPartPlusMore Reusable Coffee Filters Compatible with 1.0 and 2.0 Keurig Single Cup Coffee Maker - BPA-Free Stainless Steel Refillable K Cup Coffee Filter with Fine Mesh Screen (Pack of 4)            |0.40626391302493847|
|B0B3DB5HTC |Amazon Home             |Disposable Filters|K&J           |12 Pack Keurig Filter Replacement by K&J - Compatible with Keurig Coffee Machine (2.0 and older)                                                                                                        |0.3411060503294809 |
|B092LLM7H3 |Grocery                 |Reusable Filters  |Delibru       |Reusable coffee pods for Coffee Makers (4 PACK Black and Purple)                                                                                                                                        |0.3389491204773901 |
|B0BHNSLKNZ |Amazon Home             |Permanent Filters |Cuisinart     |Cuisinart Replacement Water Filters, 2-Pack                                                                                                                                                             |0.15453668204547413|
|B09YRPT4Q2 |Amazon Home             |Water Filters     |BELVITA       |BELVITA ADQ747935 Water Filter Replacement,Compatible with LT1000P,LFXS26973S,LMXS28626S,LMXS30796S,LMXC23796S,Kenmore Elite 9980 ADQ74793501 MDJ64844601,3 Pack                                        |0.11837583414486465|
|B0045LLC7K |Appliances              |Water Filters     |Frigidaire    |Frigidaire WF3CB Puresource3 Refrigerator Water Filter , White, 1 Count (Pack of 1)                                                                                                                     |0.10011775415853462|
|B01IAFNZGC |Tools & Home Improvement|Water Filters     |EXCELPURE     |EXCELPURE 5231JA2006A Replacement for LG LT600P,5231JA2006B, Kenmore 46-9990, 5231JA2006F,R-9990, 5231JA2006E, LFX25975ST, LFX25960ST, EFF-6003A, LFX23961ST, SGF-LB60, Refrigerator Water Filter, 3PACK|0.07943687604149981|
|B0BC65XJLJ |Amazon Home             |Water Filters     |KASTORE F1    |W10295370A Water FiIter Cap Replacement, Compatible with EDR1RXD1 Refrigerator Water FiIter 1 46-9081, 46-9930 Water FiIter Cap Replacement 1, 3Packs                                                   |0.05484415854521352|
|B07S9DJ2S2 |Amazon Home             |Ice Makers        |Amazon Renewed|Frigidaire Portable Compact Maker, Counter Top Ice Making Machine, 26lb per day (Blue) (EFIC108-BLUE) (Renewed)                                                                                         |0.05243582251940254|



### Option B: Similar to text query
Find similar items using a vectorized text query (e.g., product description or search keywords)

- Sample keyword search query =  ***'K-Cup Reusable Coffee Filter'***

**Top 10 Products similar to query:**

|parent_asin|main_category           |subcategory       |maker         |title                                                                                                                                                                                                   |distance           |
|-----------|------------------------|------------------|--------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------|
|B000DLB2FI |Amazon Home             |Reusable Filters  |Keurig Kitchenware|Keurig My K-Cup Reusable Coffee Filter - Old Model                                                                                                                                                      |0.6124652823568978 |
|B07P3Y8JWY |Amazon Home             |Reusable Filters  |Simple Cups   |Disposable Paper Coffee Filters 600 count - Compatible with Keurig, K-Cup machines & other Single Serve Coffee Brewer Reusable K Cups - Use Your Own Coffee & Make Your Own Pods - Works with All Brands|0.5178580622168953 |
|B092LLM7H3 |Grocery                 |Reusable Filters  |Delibru       |Reusable coffee pods for Coffee Makers (4 PACK Black and Purple)                                                                                                                                        |0.48147325315660827|
|B07RNJY499 |Amazon Home             |Reusable Filters  |iPartsPlusMore|iPartPlusMore Reusable Coffee Filters Compatible with 1.0 and 2.0 Keurig Single Cup Coffee Maker - BPA-Free Stainless Steel Refillable K Cup Coffee Filter with Fine Mesh Screen (Pack of 4)            |0.44580092914805713|
|B0B3DB5HTC |Amazon Home             |Disposable Filters|K&J           |12 Pack Keurig Filter Replacement by K&J - Compatible with Keurig Coffee Machine (2.0 and older)                                                                                                        |0.1796195100888521 |
|B0BHNSLKNZ |Amazon Home             |Permanent Filters |Cuisinart     |Cuisinart Replacement Water Filters, 2-Pack                                                                                                                                                             |0.12265359099238245|
|B0045LLC7K |Appliances              |Water Filters     |Frigidaire    |Frigidaire WF3CB Puresource3 Refrigerator Water Filter , White, 1 Count (Pack of 1)                                                                                                                     |0.11482036702444197|
|B09YRPT4Q2 |Amazon Home             |Water Filters     |BELVITA       |BELVITA ADQ747935 Water Filter Replacement,Compatible with LT1000P,LFXS26973S,LMXS28626S,LMXS30796S,LMXC23796S,Kenmore Elite 9980 ADQ74793501 MDJ64844601,3 Pack                                        |0.09395323461334498|
|B01IAFNZGC |Tools & Home Improvement|Water Filters     |EXCELPURE     |EXCELPURE 5231JA2006A Replacement for LG LT600P,5231JA2006B, Kenmore 46-9990, 5231JA2006F,R-9990, 5231JA2006E, LFX25975ST, LFX25960ST, EFF-6003A, LFX23961ST, SGF-LB60, Refrigerator Water Filter, 3PACK|0.07949516104560764|
|B0BD2MT2FN |Amazon Home             |Milk Frothing Pitchers|CACAKEE       |CACAKEE Milk Frothing Pitcher, 12 OZ/350ML Stainless Steel Espresso Steaming Pitchers, Coffee Milk Frother Jug for Espresso Machines Cappuccino Latte Art, Pour Cup                                     |0.046603226414400775|


## 6. Collaborative Filtering

- **Thresholds**: Set minimum ratings for products (5) and users (2) to reduce noise.

### Item-Based CF: 
Recommends items similar to those a user has rated.
- Sample ASIN ID rated by user: ***B000DLB2FI***

**Top 10 Recommended Products from Similar Users:**

|asin      |main category|subcategory       |title                                                                                                                                                                                                   |distance          |
|----------|-------------|------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------|
|B00BUFJBQS|Amazon Home  |Reusable Filters  |Disposable Paper Coffee Filters 600 count - Compatible with Keurig, K-Cup machines & other Single Serve Coffee Brewer Reusable K Cups - Use Your Own Coffee & Make Your Own Pods - Works with All Brands|0.9828166264927277|
|B0001IRRLQ|Amazon Home  |Permanent Filters |Cuisinart Replacement Water Filters, 2-Pack                                                                                                                                                             |1.0               |
|B0001IRRLG|Amazon Home  |Permanent Filters |Cuisinart Replacement Water Filters, 2-Pack                                                                                                                                                             |1.0               |
|B000TETMVK|Amazon Home  |Permanent Filters |Cuisinart Replacement Water Filters, 2-Pack                                                                                                                                                             |1.0               |
|B0045LLC7K|Appliances   |Water Filters     |Frigidaire WF3CB Puresource3 Refrigerator Water Filter , White, 1 Count (Pack of 1)                                                                                                                     |1.0               |
|B00888OEQW|Amazon Home  |Reusable Filters  |Disposable Paper Coffee Filters 600 count - Compatible with Keurig, K-Cup machines & other Single Serve Coffee Brewer Reusable K Cups - Use Your Own Coffee & Make Your Own Pods - Works with All Brands|1.0               |
|B00LGEKOMS|Amazon Home  |Reusable Filters  |iPartPlusMore Reusable Coffee Filters Compatible with 1.0 and 2.0 Keurig Single Cup Coffee Maker - BPA-Free Stainless Steel Refillable K Cup Coffee Filter with Fine Mesh Screen (Pack of 4)            |1.0               |
|B00ST3XBKG|Amazon Home  |Reusable Filters  |iPartPlusMore Reusable Coffee Filters Compatible with 1.0 and 2.0 Keurig Single Cup Coffee Maker - BPA-Free Stainless Steel Refillable K Cup Coffee Filter with Fine Mesh Screen (Pack of 4)            |1.0               |
|B01AUBYMK2|Amazon Home  |Reusable Filters  |iPartPlusMore Reusable Coffee Filters Compatible with 1.0 and 2.0 Keurig Single Cup Coffee Maker - BPA-Free Stainless Steel Refillable K Cup Coffee Filter with Fine Mesh Screen (Pack of 4)            |1.0               |
|B01DLEL4EM|Amazon Home  |Disposable Filters|12 Pack Keurig Filter Replacement by K&J - Compatible with Keurig Coffee Machine (2.0 and older)                                                                                                        |1.0               |


### User-Based CF 
  Recommends items liked by similar users.
  - Sample User ID: ***AHXVMVJEAMRUIE4FDV5ZWWPWLNCA***

***Top 10 Producst by similar users:***
  |asin      |parent_asin|main_category|subcategory       |title                                                                                                                                                                                                   |similar user_id             |score|
|----------|-----------|-------------|------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------|-----|
|B00BUFJBQS|B07P3Y8JWY |Amazon Home  |Reusable Filters  |Disposable Paper Coffee Filters 600 count - Compatible with Keurig, K-Cup machines & other Single Serve Coffee Brewer Reusable K Cups - Use Your Own Coffee & Make Your Own Pods - Works with All Brands|AE6OIQD2NCZH75MBGXG3KHHZ7VTQ|1.0  |
|B07P9KVVCS|B07P3Y8JWY |Amazon Home  |Reusable Filters  |Disposable Paper Coffee Filters 600 count - Compatible with Keurig, K-Cup machines & other Single Serve Coffee Brewer Reusable K Cups - Use Your Own Coffee & Make Your Own Pods - Works with All Brands|AE6OIQD2NCZH75MBGXG3KHHZ7VTQ|0.5  |
|B01AUBYMK2|B07RNJY499 |Amazon Home  |Reusable Filters  |iPartPlusMore Reusable Coffee Filters Compatible with 1.0 and 2.0 Keurig Single Cup Coffee Maker - BPA-Free Stainless Steel Refillable K Cup Coffee Filter with Fine Mesh Screen (Pack of 4)            |AE6JZUKDILHPDYAS4NCSW7IOVZSQ|0.5  |
|B01DLEL4EM|B0B3DB5HTC |Amazon Home  |Disposable Filters|12 Pack Keurig Filter Replacement by K&J - Compatible with Keurig Coffee Machine (2.0 and older)                                                                                                        |AE52GPMH6YF7HQMRWALACH3BBNBQ|0.5  |
|B0001IRRLQ|B0BHNSLKNZ |Amazon Home  |Permanent Filters |Cuisinart Replacement Water Filters, 2-Pack                                                                                                                                                             |AEGHRMGHNK2723AIUD3JHXYO3FQQ|0.4  |
|B00LGEKOMS|B07RNJY499 |Amazon Home  |Reusable Filters  |iPartPlusMore Reusable Coffee Filters Compatible with 1.0 and 2.0 Keurig Single Cup Coffee Maker - BPA-Free Stainless Steel Refillable K Cup Coffee Filter with Fine Mesh Screen (Pack of 4)            |AE5MCZGKGW5BCO437LRGRUHMK5RQ|0.4  |
|B0001IRRLG|B0BHNSLKNZ |Amazon Home  |Permanent Filters |Cuisinart Replacement Water Filters, 2-Pack                                                                                                                                                             |AE6HYBNWZFMX3QTIL5LY2H4AT5JQ|0.2  |
|B00888OEQW|B07P3Y8JWY |Amazon Home  |Reusable Filters  |Disposable Paper Coffee Filters 600 count - Compatible with Keurig, K-Cup machines & other Single Serve Coffee Brewer Reusable K Cups - Use Your Own Coffee & Make Your Own Pods - Works with All Brands|AEBXNYPC5RW2PRFDLQVILRSOON6Q|0.0  |
|B000TETMVK|B0BHNSLKNZ |Amazon Home  |Permanent Filters |Cuisinart Replacement Water Filters, 2-Pack                                                                                                                                                             |AERYURF52GLN35JSFZVAMRZAZA7A|0.0  |
|B00ST3XBKG|B07RNJY499 |Amazon Home  |Reusable Filters  |iPartPlusMore Reusable Coffee Filters Compatible with 1.0 and 2.0 Keurig Single Cup Coffee Maker - BPA-Free Stainless Steel Refillable K Cup Coffee Filter with Fine Mesh Screen (Pack of 4)            |AEXC6XAUJR3BCIEIFKOBV2OHPFGQ|0.0  |




### Matrix Factorization (SVD): 
Latent factor model for user-item interactions.

  - Sample User ID: ***AHXVMVJEAMRUIE4FDV5ZWWPWLNCA***

**Top 10 Product Recommendations:**

  |asin      |parent_asin|main_category|subcategory       |title                                                                                                                                                                                                   |score                 |
|----------|-----------|-------------|------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------|
|B00BUFJBQS|B07P3Y8JWY |Amazon Home  |Reusable Filters  |Disposable Paper Coffee Filters 600 count - Compatible with Keurig, K-Cup machines & other Single Serve Coffee Brewer Reusable K Cups - Use Your Own Coffee & Make Your Own Pods - Works with All Brands|3.878524360012743e-06 |
|B01AUBYMK2|B07RNJY499 |Amazon Home  |Reusable Filters  |iPartPlusMore Reusable Coffee Filters Compatible with 1.0 and 2.0 Keurig Single Cup Coffee Maker - BPA-Free Stainless Steel Refillable K Cup Coffee Filter with Fine Mesh Screen (Pack of 4)            |7.318199695606325e-07 |
|B00LGEKOMS|B07RNJY499 |Amazon Home  |Reusable Filters  |iPartPlusMore Reusable Coffee Filters Compatible with 1.0 and 2.0 Keurig Single Cup Coffee Maker - BPA-Free Stainless Steel Refillable K Cup Coffee Filter with Fine Mesh Screen (Pack of 4)            |1.678886657168029e-15 |
|B00434EMBM|B0BHNSLKNZ |Amazon Home  |Permanent Filters |Cuisinart Replacement Water Filters, 2-Pack                                                                                                                                                             |7.00615467363359e-16  |
|B000DLB2FI|B000DLB2FI |Amazon Home  |Reusable Filters  |Keurig My K-Cup Reusable Coffee Filter - Old Model                                                                                                                                                      |6.410094489115632e-16 |
|B0045LLC7K|B0045LLC7K |Appliances   |Water Filters     |Frigidaire WF3CB Puresource3 Refrigerator Water Filter , White, 1 Count (Pack of 1)                                                                                                                     |3.601035659669878e-16 |
|B08TRCBMN9|B092LLM7H3 |Grocery      |Reusable Filters  |Reusable coffee pods for Coffee Makers (4 PACK Black and Purple)                                                                                                                                        |2.1194973153256652e-16|
|B0001IRRLG|B0BHNSLKNZ |Amazon Home  |Permanent Filters |Cuisinart Replacement Water Filters, 2-Pack                                                                                                                                                             |1.7461047755239403e-16|
|B01DP1IWKU|B092LLM7H3 |Grocery      |Reusable Filters  |Reusable coffee pods for Coffee Makers (4 PACK Black and Purple)                                                                                                                                        |1.0215997104753005e-16|
|B01GAGM62M|B0B3DB5HTC |Amazon Home  |Disposable Filters|12 Pack Keurig Filter Replacement by K&J - Compatible with Keurig Coffee Machine (2.0 and older)                                                                                                        |5.974945431774294e-17 |

## 7. Evaluation


**Metrics**: \
In ranked collaborative filtering tasks like item-based or user-based recommendation, evaluation metrics like Precision@K, Recall@K, NDCG@K, and MAP@K are used to assess the quality of the ranked lists.

- **Precision@K**:\
Fraction of recommended items in the top-K that are relevant.

- **Recall@k** :\
Fraction of all relevant items that are in the top-K recommendations

- **MAP@K (Mean Average Precision)** :\
Averages the precision at each position where a relevant item occurs

- **NDCG@K (Normalized Discounted Cumulative Gain)**: \
Measures ranking quality — rewards placing relevant items higher

- **Hit Rate@K**:\
For each user, if at least one relevant item is in the top-K recommendations, it's a "hit"


- **Results** (Top-10, Leave-One-Out evaluation):
  - **Item-Based CF**: 
    - Hit Rate: ~0.75
    - Precision@10: ~0.075
    - Recall@10: ~0.75
    - MAP: ~0.30
    - NDCG@10: ~0.41
  - **User-Based CF**: 
    - Hit Rate: ~0.69
    - Precision@10: ~0.069
    - Recall@10: ~0.69
    - MAP: ~0.18
    - NDCG@10: ~0.30
  - **Matrix Factorization (SVD)**: 
    - Hit Rate: ~0.54
    - Precision@10: ~0.054
    - Recall@10: ~0.54
    - MAP: ~0.26
    - NDCG@10: ~0.32

With sparse dataset such as the amazon reviews, lower metric values are common and expected. Below metric values can be considered a good baseline. 

- HitRate@10 ~ 0.5
- Precision@10 ~ 0.08
- Recall@10 ~ 0.05
- NDCG@10 ~ 0.2
- MAP@10 ~ 0.05


**Visualization**: Comparing evaluation metrics for all algorithms. 

![ranking_metrics](reports/figures/top-k_ranking_metrics.png)

## 8. Key Findings

- **Data**: The dataset is sparse, with most users rating only a few products.
- **Category Imbalance**: A few categories dominate the ratings.
- **Recommendation Quality**: Item-based collaborative filtering performed best in this scenario, followed by user-based and SVD.
- **Cold Start**: Content-based filtering and top-rated recommendations help address the cold-start problem for new users/items.

---

For more details on code and implementation, see the full notebook: [notebooks/recsys.ipynb](notebooks/recsys.ipynb)