class EarlyStopping:
    """
    최적화 과정에서 성능 개선이 없을 때 조기 종료를 결정하는 클래스.

    이 클래스는 최대화(maximization) 문제를 가정합니다. (e.g., 샤프 비율)
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.001):
        """
        EarlyStopping 초기화.

        Args:
            patience (int): 성능 개선이 없을 때 얼마나 많은 시도를 기다릴지 결정.
            min_delta (float): 유의미한 성능 개선으로 간주할 최소 변화량.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.reset()

    def check(self, current_score: float) -> bool:
        """
        현재 점수를 바탕으로 조기 종료 여부를 확인하고 상태를 업데이트합니다.

        Args:
            current_score (float): 현재 최적화 시도에서 얻은 점수.

        Returns:
            bool: 최적화를 중단해야 하면 True, 아니면 False.
        """
        if self.should_stop:
            return True

        if current_score > self.best_score + self.min_delta:
            self.best_score = current_score
            self.wait_count = 0
        else:
            self.wait_count += 1

        if self.wait_count >= self.patience:
            self.should_stop = True

        return self.should_stop

    def reset(self) -> None:
        """
        모든 상태를 초기값으로 리셋합니다.
        """
        self.wait_count = 0
        self.best_score = float('-inf')
        self.should_stop = False 