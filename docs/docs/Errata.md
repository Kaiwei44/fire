---
title: Errata
permalink: /errata/
nav_order: 100
---
# Errata for *Navigating the Factor Zoo: The Science of Quantitative Investing*
---
## Overview

This document captures all **verified printing and content errors** identified in *_Navigating the Factor Zoo: The Science of Quantitative Investing_*. It is maintained in the Fire Institute GitHub repository (https://github.com/fire-institute/fire) under `docs/docs/errata.md`.

### Structure of Entries
Each erratum follows this format:

| Field              | Description                                                        |
| ------------------ | ------------------------------------------------------------------ |
| **Anchor**         | Unique Markdown heading used as the link target.                   |
| **Original**       | Verbatim the incorrect text, caption, or equation.                 |
| **Correction**     | The accurate replacement text, caption, or equation.               |
| **Note**           | (Optional) Additional context or explanation.                      |

---

### Submitting a New Error Report
To contribute:

1. **Search** existing GitHub issues to avoid duplicates.  
2. **Open a new issue** with the title:
   ```
   [Errata] Page <number> – brief description
   ```
3. **Fill in the template** in the issue body:
   ```markdown
   **Page**:  
   **Section or Heading**:  
   **Original**:  
   **Correction**:  
   ```
4. A maintainer will review, label it **confirmed**, and then add it here.

---

## Table of Content

* [First Edition — Routledge (Hardcover & Paperback)](#first-edition-routledge-hardcover--paperback)
  * [Page 66 – Equation 3.19](#page-66-equation-3-19)



---

## First Edition — Routledge (Hardcover & Paperback) <a name="first-edition-routledge-hardcover--paperback"></a>

- **Publisher**: Routledge  
- **Publication Date**: November 20, 2024 (Hardcover) / December 9, 2024 (Paperback)  
- **Formats**: Hardcover (296 pp.) / Paperback (310 pp.)  
- **ISBN-10**: 1032768436 (HC) / 103276841X (PB)  
- **ISBN-13**: 978-1032768434 (HC) / 978-1032768410 (PB)  

### Page 66 – Equation 3.19 <a name="page-66-equation-3-19"></a>
**Original** 

> In the limit of $n \rightarrow \infty$，$R V_{t}^{+} \rightarrow \text{ }_{t-1}^{t} \sigma_{s}^{2} ds+\sum_{t-1 \leq \tau \leq t} J_{\tau J_{\tau}>0 }^{2} $, $ R V_{t}^{-} \rightarrow \int_{t-1}^{t} \sum_{s}^{2} d s+\sum_{t-1 \leq \tau \leq t} J_{\tau J_{\tau}0 }^{2}  $, and,
>
> $$S J_{t}=\sum_{t-1 \leq \tau \leq t} J_{\tau J_{\tau}>0 }^{2} -\sum_{t-1 \leq \tau \leq t} J_{\tau J_{\tau} 0}^{2} $$ 

**Correction** 

> In the limit of $n\to \infty$, $RV_t^+ \to \int _{t- 1}^t\sigma _s^2ds+ \sum_{t- 1\leq \tau \leq t}J_\tau^2 \mathbb{I} _{J_\tau > 0}$, $RV_t^- \to \int_{t- 1}^t \sigma_s^2 ds + \sum_{t-1\leq\tau\leq t}J_\tau^2\mathbb{I}_{J_\tau<0} $, and, 
>
> $$SJ_t = \sum_{t- 1\leq \tau \leq t}J_\tau^2 \mathbb{I} _{J_\tau > 0}-\sum_{t-1\leq\tau\leq t}J_\tau^2\mathbb{I}_{J_\tau<0}.$$

**Note** 

> Inserted the missing integral symbol, properly representing the continuous term as $\int_{t-1}^t\sigma_s^2\,ds$. Replaced the ambiguous jump‐index notation with indicator functions $\mathbb{I}_{J_\tau>0}$ and $\mathbb{I}_{J_\tau<0}$ to clearly separate positive and negative jumps.

