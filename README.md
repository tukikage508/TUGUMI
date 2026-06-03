# TUGUMI - Self-Autonomous AI Agent Framework

自律型AIエージェント。LLMとLangGraphを使用して、複雑なタスクを自動的に実行・学習できるフレームワーク。

## 機能

- **自律実行**: ユーザーのゴールを分析し、計画を立ててステップバイステップで実行
- **ツール統合**: Web検索、Webスクレイピング、コマンド実行、ファイル保存
- **エラーハンドリング**: 失敗時の自動リトライ、エラーパターンの学習
- **永続的メモリ**: 実行履歴、ツール効果測定、エラー解決策のキャッシング
- **LLMサーバー連携**: llama.cpp等のローカルLLMサーバーと通信
- **進捗監視**: バックグラウンドヘルスチェック、詳細ログ

## セットアップ

### 1. 依存ライブラリのインストール

```bash
pip install -r requirements.txt
```

### 2. LLMサーバーの起動

llama.cpp等のローカルLLMサーバーを起動してください。デフォルトは `http://127.0.0.1:8080` です。

#### llama.cpp を使用する場合（推奨）:

```bash
# llama.cppをビルド・実行（Modelファイルが必要）
./main -m model.gguf -c 2048 --host 127.0.0.1 --port 8080
```

または Python APIサーバー:

```bash
pip install llama-cpp-python
python -m llama_cpp.server --model_path model.gguf --host 127.0.0.1 --port 8080
```

## 使用方法

### 基本的な使用

```bash
# 基本的なタスク実行
python main.py "Create a Python script that prints Hello World"

# カスタムLLMサーバーを指定
python main.py -u http://localhost:8000 "Your task here"

# タイムアウトを設定（秒単位）
python main.py --timeout 600 "Long-running task"

# エージェントの状態を表示
python main.py --status
```

### Pythonコードでの使用

```python
from src.runner import AgentRunner

# 初期化
runner = AgentRunner(
    llm_server_url="http://127.0.0.1:8080",
    llm_timeout=300,
    memory_file="data/memory.json"
)

# タスク実行
result = runner.run_task("Search for Python best practices and save results")

if result["success"]:
    print("Task completed!")
    print(f"Duration: {result['summary']['duration_seconds']}s")
    print(f"Tools used: {result['summary']['tools_used']}")
else:
    print(f"Task failed: {result['error']}")

# クリーンアップ
runner.cleanup()
```

## アーキテクチャ

```
┌─────────────────────────────────────────────────────┐
│                    main.py (CLI)                    │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│              AgentRunner                            │
│  ├─ LLMClient (health monitoring)                   │
│  ├─ TaskMemory (persistent learning)                │
│  └─ TUGUMIGraph (execution engine)                  │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│         LangGraph State Machine                     │
│  1. Understand Goal → 2. Plan → 3. Execute         │
│  4. Evaluate → 5. Learn/Adapt → 6. Complete        │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│            Tool Executor                            │
│  ├─ web_search (DuckDuckGo)                        │
│  ├─ fetch_and_summarize (BeautifulSoup)            │
│  ├─ execute_command (subprocess)                    │
│  ├─ install_package (pip)                          │
│  └─ save_output (file write)                       │
└─────────────────────────────────────────────────────┘
```

## ファイル構成

```
TUGUMI/
├── main.py                 # エントリーポイント
├── requirements.txt        # 依存ライブラリ
├── README.md              # このファイル
├── src/
│   ├── __init__.py        # パッケージ初期化
│   ├── agent_state.py     # エージェント状態管理
│   ├── graph.py           # LangGraph実行エンジン
│   ├── runner.py          # タスク実行エンジン
│   ├── tools.py           # ツール実行
│   ├── llm_client.py      # LLMサーバー通信
│   ├── logger.py          # ログシステム
│   └── memory.py          # 永続メモリシステム
├── logs/                  # ログディレクトリ（自動生成）
└── data/
    └── memory.json        # メモリファイル（自動生成）
```

## タスク実行フロー

1. **Goal Understanding**: LLMに目標を分析させ、必要なリソースと成功基準を特定
2. **Planning**: 実行計画（ステップ）を生成
3. **Execution**: 各ステップを実行し、必要に応じてツールを呼び出し
4. **Evaluation**: 進捗を評価し、成功/失敗/停滞を判定
5. **Learning**: 失敗時はエラーパターンから学習し、計画を適応
6. **Adaptation**: 新しい戦略で再実行
7. **Completion**: 完了またはリトライ上限に達した場合は終了

## ツール一覧

### web_search
インターネット検索（DuckDuckGo）
```python
result = tools.web_search("query", max_results=5)
# Returns: {"success": bool, "results": [...], "count": int}
```

### fetch_and_summarize
Webページの内容取得とテキスト抽出
```python
result = tools.fetch_and_summarize("https://example.com")
# Returns: {"success": bool, "title": str, "content": str}
```

### execute_command
シェルコマンド実行
```python
result = tools.execute_command("ls -la", timeout=60)
# Returns: {"success": bool, "stdout": str, "stderr": str, "return_code": int}
```

### install_package
Pythonパッケージのインストール
```python
result = tools.install_package("requests")
# Returns: {"success": bool, ...}
```

### save_output
ファイルへの保存
```python
result = tools.save_output("content", "filename.txt")
# Returns: {"success": bool, "path": str, "size": int}
```

## 設定

### 環境変数（.envファイル）

```env
LLM_SERVER_URL=http://127.0.0.1:8080
LLM_TIMEOUT=300
MEMORY_FILE=data/memory.json
```

### ランタイムパラメータ

`AgentRunner`初期化時に以下をカスタマイズ可能:

- `llm_server_url`: LLMサーバーのURL
- `llm_timeout`: LLMリクエストのタイムアウト（秒）
- `memory_file`: メモリファイルのパス
- `max_reconnect_attempts`: LLM再接続試行回数

## ログ

ログは `logs/` ディレクトリに自動保存されます。

```
logs/
├── TUGUMI_20240101_120000.log
├── TUGUMI_MAIN_20240101_120100.log
└── ...
```

## トラブルシューティング

### LLMサーバーに接続できない

```
Error: LLM server is not available
```

**解決策:**
1. LLMサーバーが起動しているか確認
2. URLとポートが正しいか確認
3. ファイアウォールが接続をブロックしていないか確認

### メモリ不足エラー

**解決策:**
- `max_items` を減らす: `TaskMemory(max_items=50)`
- ディスク容量を確認

### タイムアウトエラー

```
Error: Request timeout after 300s
```

**解決策:**
- `--timeout` を増やす: `python main.py --timeout 600 "task"`
- LLMモデルが大きすぎないか確認
- CPU/メモリ使用率を確認

## パフォーマンス最適化

1. **キャッシング**: 前回のソリューションをキャッシュして再利用
2. **ツール効果測定**: 各ツールの成功率と実行時間を追跡
3. **エラー学習**: エラーパターンから解決策を検索・保存
4. **バックグラウンド監視**: LLMのヘルスチェックをバックグラウンドで実行

## 学習・拡張

### カスタムツールの追加

`src/tools.py` に新しいメソッドを追加:

```python
def custom_tool(self, param: str) -> Dict[str, Any]:
    """Custom tool implementation"""
    try:
        # Implementation
        return {"success": True, "result": "..."}
    except Exception as e:
        return {"success": False, "error": str(e)}
```

### グラフの拡張

`src/graph.py` のノードを追加してワークフローを拡張:

```python
self.graph.add_node("custom_node", self._custom_node)
```

## ライセンス

MIT License
