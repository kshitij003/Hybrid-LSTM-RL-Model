package com.trader.trading_backend.Repository.Market_Data_Management;

import com.trader.trading_backend.entity.Market_Data_Management.Stock;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.Optional;

@Repository
public interface StockRepository extends JpaRepository<Stock,Long> {

    Optional<Stock> findByTickerIgnoreCase(String ticker);
}
