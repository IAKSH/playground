package me.iaksh.entity;

import java.math.BigDecimal;

public class YearlySales {
    private Integer year;
    private BigDecimal totalSales;

    public BigDecimal getTotalSales() {
        return totalSales;
    }

    public Integer getYear() {
        return year;
    }

    public void setTotalSales(BigDecimal totalSales) {
        this.totalSales = totalSales;
    }

    public void setYear(Integer year) {
        this.year = year;
    }
}
