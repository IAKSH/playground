package me.iaksh.entity;

import java.math.BigDecimal;
import java.sql.Date;

public class DailySales {
    private Date day;
    private BigDecimal totalSales;

    public BigDecimal getTotalSales() {
        return totalSales;
    }

    public Date getDay() {
        return day;
    }

    public void setDay(Date day) {
        this.day = day;
    }

    public void setTotalSales(BigDecimal totalSales) {
        this.totalSales = totalSales;
    }
}
