package com.trader.trading_backend.Repository.Portfolio_Trading_Engine;

import com.trader.trading_backend.entity.Market_Data_Management.Stock;
import com.trader.trading_backend.entity.Portfolio_Trading_Engine.Holding;
import com.trader.trading_backend.entity.Portfolio_Trading_Engine.Portfolio;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.Optional;

@Repository
public interface HoldingRepository extends JpaRepository<Holding,Long> {
    Optional<Holding> findByPortfolioAndStock(Portfolio portfolio, Stock stock);
}
