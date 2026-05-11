package com.trader.trading_backend.Repository.Market_Data_Management;

import com.trader.trading_backend.entity.Market_Data_Management.MarketNews;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface MarketNewsRepository extends JpaRepository<MarketNews, Long> {
    List<MarketNews> findByStockIdOrderByPublishedAtDesc(Long stockId);
}
