package com.trader.trading_backend.Repository.AI_Model_Governance;

import com.trader.trading_backend.entity.AI_Model_Governance.ModelSignal;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface ModelSignalRepository extends JpaRepository<ModelSignal,Long> {
}
