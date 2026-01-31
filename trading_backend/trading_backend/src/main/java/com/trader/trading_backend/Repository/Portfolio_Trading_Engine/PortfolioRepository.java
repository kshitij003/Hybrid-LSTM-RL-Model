package com.trader.trading_backend.Repository.Portfolio_Trading_Engine;

import com.trader.trading_backend.entity.Portfolio_Trading_Engine.Portfolio;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface PortfolioRepository extends JpaRepository<Portfolio,Long> {
}
