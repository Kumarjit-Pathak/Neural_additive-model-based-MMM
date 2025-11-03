Objective : Create MMM and Brand Equity Models to quantify the short term and long term effects of marketing activities.

Data details :

ID: The unique identification of each SKU ID: The unique identification of each SKU Order ,  The unique identification number of each order Order item ID: Suppose you order 2 different products under the same order, it generates 2 different order Item IDs under the same order ID; orders are tracked by the Order Item ID GMV: Gross Merchandise Value or Revenue Units: Number of units of the specific product sold SLA: Number of days it typically takes to deliver the product MRP: Maximum retail price of the product Procurement SLA: Time typically taken to procure the product
Order ID: The unique identification number of each order
Order item ID: Suppose you order 2 different products under the same order, it generates 2 different order Item IDs under the same order ID; orders are tracked by the Order Item ID
GMV: Gross Merchandise Value or Revenue
Units: Number of units of the specific product sold
SLA: Number of days it typically takes to deliver the product
MRP: Maximum retail price of the product
Procurement SLA: Time typically taken to procure the product

---

## Comprehensive Data Understanding

### Dataset Overview

The data folder contains 7 datasets covering e-commerce sales, marketing investments, and brand equity metrics for an electronics retailer from July 2015 to June 2016.

### 1. Sales.csv (112.1 MB)
**Description**: Transaction-level order data with granular product and delivery information.

**Time Range**: October 2015 - June 2016

**Key Fields**:
- ID: Product SKU identifier
- Date: Order date and time
- ID_Order: Unique order identifier
- ID_Item_ordered: Unique order item identifier (multiple items per order possible)
- GMV: Gross Merchandise Value (Revenue per transaction)
- Units_sold: Quantity of products sold
- SLA: Service Level Agreement (days to deliver)
- MRP: Maximum Retail Price
- Procurement_SLA: Procurement lead time
- Product_Category: Main category (CE - Consumer Electronics)
- Analytic_Category: Sub-category (Camera, CameraAccessory)
- Sub_category: Detailed category
- product_analytic_vertical: Product type (e.g., CameraTripod, Lens)

**Product Categories**: Camera, CameraAccessory, and CE (Consumer Electronics)

**Data Characteristics**:
- Contains individual transaction records
- Some GMV values are missing (blank cells)
- Includes timestamp data for temporal analysis

### 2. firstfile.csv (149.2 MB)
**Description**: Daily transaction-level sales data with promotion tracking and detailed product hierarchies.

**Time Range**: July 2015 onwards

**Key Fields**:
- Date: Transaction date
- Sales_name: Promotion type ("No Promotion" or specific promotion name)
- gmv_new: Revenue per transaction
- units: Quantity sold
- product_mrp: Maximum Retail Price
- discount: Discount amount (MRP - GMV)
- product_category: Main category
- product_subcategory: Mid-level category
- product_vertical: Specific product type

**Product Categories**:
- Camera
- CameraAccessory
- EntertainmentSmall (Speakers, Audio/Video products)
- GameCDDVD (Physical and digital games)
- GamingHardware (Consoles, accessories)

**Data Characteristics**:
- Most granular sales dataset
- Links transactions to promotional events
- Contains complete discount information

### 3. Secondfile.csv (Aggregated Monthly Data)
**Description**: Monthly aggregated sales and marketing data combining all key metrics for analysis.

**Time Range**: July 2015 - June 2016 (12 months)

**Key Metrics by Category**:
- Revenue: Revenue_Camera, Revenue_CameraAccessory, Revenue_EntertainmentSmall, Revenue_GameCDDVD, Revenue_GamingHardware
- Units: Units_Camera, Units_CameraAccessory, Units_EntertainmentSmall, Units_GameCDDVD, Units_GamingHardware
- MRP: Mrp_Camera, Mrp_CameraAccessory, Mrp_EntertainmentSmall, Mrp_GameCDDVD, Mrp_GamingHardware
- Discount: Discount_Camera, Discount_CameraAccessory, Discount_EntertainmentSmall, Discount_GameCDDVD, Discount_GamingHardware
- Totals: total_gmv, total_Units, total_Mrp, total_Discount

**Marketing Investment Channels** (from MediaInvestment.csv):
- TV, Digital, Sponsorship, Content.Marketing, Online.marketing, Affiliates, SEM, Radio, Other
- Total.Investment: Sum of all marketing spend

**Brand Equity Metrics**:
- NPS: Monthly Net Promoter Score

**Data Characteristics**:
- Ready-to-use for Marketing Mix Modeling
- All data sources integrated at monthly level
- Enables direct analysis of marketing ROI and brand impact

### 4. MediaInvestment.csv
**Description**: Monthly marketing spend across multiple channels.

**Time Range**: July 2015 - June 2016

**Fields**:
- Year, Month: Time identifiers
- Total Investment: Total marketing spend for the month
- Channel breakdown: TV, Digital, Sponsorship, Content Marketing, Online marketing, Affiliates, SEM, Radio, Other

**Investment Ranges**:
- Minimum: 5.1M (Aug 2015)
- Maximum: 170.2M (Oct 2015)
- Average: ~75-80M per month

**Data Characteristics**:
- Some channels have missing data (Radio, Other) in certain months
- High variability in monthly spend
- Major spikes during festival seasons (Oct-Nov 2015, Dec 2015)

### 5. MonthlyNPSscore.csv
**Description**: Monthly Net Promoter Score measuring customer satisfaction and brand equity.

**Time Range**: July 2015 - June 2016

**Fields**:
- Date: First day of each month
- NPS: Net Promoter Score

**NPS Score Range**:
- Minimum: 44.4 (Oct 2015)
- Maximum: 60 (Aug 2015)
- Average: ~49.6

**Data Characteristics**:
- Monthly frequency aligned with other datasets
- Shows brand health fluctuation
- Can be used as dependent or independent variable in modeling

### 6. ProductList.csv
**Description**: Product catalog with overall sales performance statistics.

**Fields**:
- Product: Product vertical name
- Frequency: Total units sold (across entire period)
- Percent: Percentage of total sales volume

**Top 10 Products by Volume**:
1. LaptopSpeaker: 287,850 units (17.5%)
2. MobileSpeaker: 250,250 units (15.2%)
3. AudioMP3Player: 112,892 units (6.8%)
4. PhysicalGame: 105,061 units (6.4%)
5. HomeAudioSpeaker: 85,607 units (5.2%)
6. GamingHeadset: 62,311 units (3.8%)
7. GamePad: 59,115 units (3.6%)
8. DSLR: 56,615 units (3.4%)
9. Flash: 48,769 units (3.0%)
10. SelectorBox: 46,253 units (2.8%)

**Total**: 1,648,824 units across 75 product types

**Data Characteristics**:
- Complete product taxonomy
- Shows long-tail distribution (many products with <1% share)
- Includes "\N" category (5,828 units, 0.4%) indicating missing product data

### 7. SpecialSale.csv
**Description**: Calendar of promotional events and sales campaigns.

**Time Range**: July 2015 - May 2016

**Major Promotional Events**:
- Eid & Rathayatra sale (Jul 2015)
- Independence Sale (Aug 2015)
- Rakshabandhan Sale (Aug 2015)
- Daussera sale (Oct 2015)
- Big Diwali Sale (Nov 2015) - 8 days
- Christmas & New Year Sale (Dec 2015 - Jan 2016) - 10 days
- Republic Day (Jan 2016)
- BED, FHSD (Feb 2016)
- Valentine's Day (Feb 2016)
- BSD-5 (Mar 2016)
- Pacman (May 2016)

**Data Characteristics**:
- Daily-level promotional calendar
- Most sales run 2-10 days
- Aligned with Indian festivals and national holidays
- Can be used to create promotional dummy variables

---

## Data Relationships and Hierarchy

```
Transaction Level:
├── Sales.csv (Oct 2015 - Jun 2016)
└── firstfile.csv (Jul 2015 onwards)

Aggregated Level:
└── Secondfile.csv (Monthly aggregated)
    ├── Aggregates from: Sales.csv + firstfile.csv
    ├── Joined with: MediaInvestment.csv
    ├── Joined with: MonthlyNPSscore.csv
    └── Enhanced by: SpecialSale.csv

Reference Data:
└── ProductList.csv (Product master with performance stats)
```

---

## Product Category Hierarchy

**Level 1: Product Category**
- Camera
- CameraAccessory
- EntertainmentSmall
- GameCDDVD
- GamingHardware
- CE (Consumer Electronics)

**Level 2: Product Subcategory** (Examples)
- Camera: Camera
- CameraAccessory: CameraAccessory, CameraStorage
- EntertainmentSmall: Speaker, HomeAudio, AudioMP3Player, TVVideoSmall
- GameCDDVD: Game
- GamingHardware: GamingAccessory

**Level 3: Product Vertical** (Examples)
- DSLR, Point & Shoot, Sports & Action
- CameraTripod, CameraBattery, Lens, Flash, CameraBag
- LaptopSpeaker, MobileSpeaker, HomeAudioSpeaker, RemoteControl
- PhysicalGame, CodeInTheBoxGame
- GamePad, GamingHeadset, GamingMouse, GamingKeyboard

---

## Marketing Channels

**Paid Media**:
- TV: Television advertising
- Digital: Digital advertising (display, video, social)
- SEM: Search Engine Marketing (paid search)
- Radio: Radio advertising

**Owned & Partnership**:
- Sponsorship: Event and sports sponsorships
- Content Marketing: Content creation and distribution
- Online marketing: Owned digital properties
- Affiliates: Affiliate marketing partnerships
- Other: Miscellaneous marketing activities

---

## Key Metrics Definition

**Revenue Metrics**:
- GMV (Gross Merchandise Value): Total revenue from product sales
- MRP: Maximum Retail Price (list price)
- Discount: Price reduction (MRP - GMV)

**Volume Metrics**:
- Units: Number of products sold
- Frequency: Total units sold for a product across the period

**Operational Metrics**:
- SLA: Service Level Agreement (delivery time in days)
- Procurement_SLA: Time to procure product (inventory lead time)

**Marketing Metrics**:
- Total Investment: Sum of all marketing channel spend
- Channel Spend: Individual channel investment amounts

**Brand Equity Metrics**:
- NPS: Net Promoter Score (customer satisfaction and loyalty)

---

## Analysis Opportunities

### 1. Marketing Mix Modeling (MMM)
- **Dependent Variables**: Revenue (total or by category), Units sold
- **Marketing Variables**: TV, Digital, SEM, Sponsorship, Affiliates, etc.
- **Control Variables**: Seasonality, promotions, pricing (discount %), NPS
- **Modeling Approach**: Neural Additive Models to capture non-linear effects and interactions

### 2. Brand Equity Analysis
- **Dependent Variable**: NPS score
- **Marketing Variables**: Above-the-line (TV, Sponsorship) vs. below-the-line (SEM, Affiliates)
- **Control Variables**: Sales performance, pricing strategy

### 3. Short-term vs. Long-term Effects
- **Short-term**: Immediate impact of promotions and performance marketing (SEM, Affiliates)
- **Long-term**: Sustained impact of brand-building activities (TV, Sponsorship, Content Marketing) on NPS and baseline sales

### 4. Channel Attribution
- ROI calculation for each marketing channel
- Diminishing returns analysis
- Optimal budget allocation

### 5. Promotional Effectiveness
- Impact of different festival sales on revenue and volume
- Discount elasticity by product category
- Halo effects across product categories

### 6. Seasonality Patterns
- Monthly trends and year-over-year growth
- Festival season performance (Diwali, Christmas)
- Category-specific seasonal patterns

---

## Data Quality Notes

**Strengths**:
- Comprehensive coverage across sales, marketing, and brand metrics
- Multiple granularity levels (daily transactions + monthly aggregates)
- 12-month period captures full seasonal cycle
- Rich product hierarchy enables category-level analysis

**Limitations**:
- Some missing values in Sales.csv (GMV field)
- MediaInvestment.csv has gaps in Radio and Other channels
- Short time series (12 months) may limit long-term trend analysis
- firstfile.csv and Sales.csv have different time coverage periods

**Recommendations**:
- Use Secondfile.csv as primary dataset for MMM (clean, aggregated, complete)
- Use firstfile.csv/Sales.csv for granular analysis and validation
- Handle missing marketing channel data through imputation or exclusion
- Create lagged variables to capture delayed marketing effects