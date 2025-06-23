---
title: Data Management
permalink: /data/
nav_order: 4
---

# Data Management

Fire provides a user-friendly interface for downloading and managing financial data. By leveraging the pre-cleaned and processed data pipeline from the Fire Institute, you can focus more on research and modeling rather than data preparation.


Currently, Fire only porvides the data from the Chinese A stock market. We will provide more data in the future.


# Download Data

We provide a simple command-line interface to download the data. You can use the following command to download the data:

```bash
firefin download
```

This command will download the latest data from the Fire Institute and store it in the `~/.fire/data/raw` directory, all data
will be organized in feather format. (Maybe we will consider other database or k-v store in the future) cause we do not update the data frequently, so we choose feather format for its fast read/write speed.

# Load Data

If you have downloaded the data manually or received it from another source, you can use the following command to load it into the Firefin system:

```bash
firefin load <file_path>
```

This command will extract the contents of the provided tar file and place them in the appropriate directory within the Firefin system.

# Data Structure

The data is organized in a structured format to facilitate easy access and manipulation. Here is an overview of the data structure:


| Date       | security1  | security2  | security2  | ...  | securityN  |   
|------------|------------|------------|------------|------|------------|
| 2023-01-01 | 10.5   | 10.7   | 10.8   | ...  | 10.9   |   
| 2023-01-02 | 10.6   | 10.8   | 10.9   | ...  | 11.1  |   
| ...        | ...    | ...    | ...    | ...  |
| 2023-12-31 | 11.0   | 11.2   | 11.3   | ...  | 11.4   |   


1. ALL data is stored in a single Feather file named `data_name.feather`.
2. Each row represents a date.
3. Each column represents a security, identified by its ticker symbol.
4. The values in the cells represent the closing prices of the securities on the corresponding dates.
5. **index(date) and columns(securities) are exactly the same across A datasets.** For example, 'A-share chinese market'

With the above structure, you can easily perform time-series analysis, portfolio optimization, and other financial analyses, with out thinking about the data alignment issue.

## Current Data Structure Limitations

1. Inclusion of Delisted Securities: The dataset retains entries for delisted securities, though their corresponding fields are populated with zeroes. While this poses limited disruption due to the low incidence of delistings in the A-share market, it may introduce unnecessary complexity for users unaware of this convention.
2. Unstructured Security Sequencing: The "Securities" column does not follow a predetermined order (e.g., listing date or ticker sequence). This lack of inherent sorting logic necessitates external reference to the ListingData index for proper chronological organization.
3. Ambiguous Naming Conventions: The dataset’s frame naming relies on user-defined labels, which can lead to confusion when shared across teams or workflows. Standardized naming protocols would enhance interpretability and maintain consistency with industry conventions.


# More On Data Structure

Currently, we have two types of data structure:

1. `pd.DataFrame` stand for (Time × Stock)
2. `pd.Series` stand for Time Series for market variable

which is enough for most of the cases. But for transaction data or order by order data, we have no choice but to use `pd.DataFrame` to store the data, but index have to be (Time × Stock) and columns should be the variable name. 

If the data is periodic in time, but may not be float/int/bool, 