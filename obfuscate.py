"""
obfuscate.py — 将 docs/core.js 混淆为 docs/core.min.js
用法: python obfuscate.py

保留公共 API 名称（UI 代码需要调用），混淆内部实现。
"""
import re, os

SRC = os.path.join(os.path.dirname(__file__), 'docs', 'core.js')
DST = os.path.join(os.path.dirname(__file__), 'docs', 'core.min.js')

# 内部函数/方法名 → 混淆名映射
# 注意：只重命名UI代码不直接调用的内部实现
RENAME_MAP = {
    # Cloud helpers (内部名，通过 window.xxx 导出原名供 UI 使用)
    'cloudDocRef':        '_$k',
    'cloudPushState':     '_$l',
    'cloudPushHistory':   '_$m',
    'cloudPushMachineList': '_$n',
    'cloudDeleteMachine': '_$o',
    'cloudPullAll':       '_$p',
    'cloudPushAll':       '_$q',
    'setSyncStatus':      '_$r',
    'getMachinesLocal':   '_$s',
    'saveMachinesLocal':  '_$t',
    # Debounce 内部实现
    '_debounce':              '_$u',
    '_cloudPushStateNow':     '_$v',
    '_cloudPushHistoryNow':   '_$w',
    '_debouncedPushState':    '_$x',
    '_debouncedPushHistory':  '_$y',
    # PinballStrategy 仅内部调用的私有方法
    '_betForMarbles':     '_$f',
    '_betForCards':       '_$g',
}

# 以下名称 UI 代码直接调用，绝不能重命名:
# PinballStrategy 类方法: getLandingProbs, recordLanding, winProbability,
#   optimalBet, recommend, expectedValueTable, _save, load
# recommend() 返回对象属性: winProbability, optimalBet, expectedMarbleReturn,
#   expectedScoreCards, marbleRoi
# expectedValueTable() 返回对象属性: cardOptimalBet


def strip_comments(code):
    """移除 // 行注释和 /* */ 块注释"""
    # 块注释
    code = re.sub(r'/\*[\s\S]*?\*/', '', code)
    # 行注释 (注意不要误伤字符串中的 //)
    code = re.sub(r'(?<!["\':])//[^\n]*', '', code)
    return code


def minify(code):
    """压缩空白"""
    lines = code.split('\n')
    result = []
    for line in lines:
        stripped = line.strip()
        if stripped:
            result.append(stripped)
    code = '\n'.join(result)
    # 合并多余空行
    code = re.sub(r'\n{2,}', '\n', code)

    # 在字符串外压缩空白:
    # 先将字符串提取保护，压缩空白后再放回
    strings = []

    def preserve_string(m):
        strings.append(m.group(0))
        return f'__STR{len(strings)-1}__'

    # 保护字符串字面量和模板字面量
    protected = re.sub(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"|`(?:[^`\\]|\\.)*`", preserve_string, code)

    # 压缩多余空格为单个空格
    protected = re.sub(r'[ \t]+', ' ', protected)
    # 移除运算符周围的空格（关键字与标识符之间的空格自动保留）
    protected = re.sub(r' ?([{};,=<>!&|?:+\-*/()[\]]) ?', r'\1', protected)

    # 恢复字符串
    for i, s in enumerate(strings):
        protected = protected.replace(f'__STR{i}__', s)

    return protected


def apply_renames(code):
    """替换内部标识符名称"""
    for orig, obf in RENAME_MAP.items():
        # 使用单词边界替换
        code = re.sub(r'\b' + re.escape(orig) + r'\b', obf, code)
    return code


def create_wrapper(code):
    """
    包裹为 IIFE + 将公共 API 导出到 window
    这样源码在 IIFE 内不直接可读，但公共 API 仍然可用
    """
    # 收集需要导出的名称
    exports = []

    # 常量
    for name in ['NUM_SLOTS', 'MIN_BET', 'MAX_BET', 'MULT_SLOTS',
                 'LIMIT_OFFLINE', 'LIMIT_LOGIN', 'LIMIT_ACTIVATED']:
        exports.append(f'window.{name}={name};')

    # Firebase 相关 (需要可写)
    exports.append('window.FIREBASE_CONFIG=FIREBASE_CONFIG;')
    exports.append('Object.defineProperty(window,"fbReady",{get(){return fbReady},set(v){fbReady=v}});')
    exports.append('Object.defineProperty(window,"fbAuth",{get(){return fbAuth},set(v){fbAuth=v}});')
    exports.append('Object.defineProperty(window,"fbDb",{get(){return fbDb},set(v){fbDb=v}});')
    exports.append('Object.defineProperty(window,"cloudUid",{get(){return cloudUid},set(v){cloudUid=v}});')

    # Cloud 函数 (UI 代码通过原始名称调用)
    cloud_fns = [
        ('cloudDocRef', '_$k'), ('cloudPushState', '_$l'),
        ('cloudPushHistory', '_$m'), ('cloudPushMachineList', '_$n'),
        ('cloudDeleteMachine', '_$o'), ('cloudPullAll', '_$p'),
        ('cloudPushAll', '_$q'), ('setSyncStatus', '_$r'),
        ('getMachinesLocal', '_$s'), ('saveMachinesLocal', '_$t'),
    ]
    for pub, obf in cloud_fns:
        exports.append(f'window.{pub}={obf};')

    # PinballStrategy class
    exports.append('window.PinballStrategy=PinballStrategy;')

    export_block = '\n'.join(exports)

    return f'(function(){{\n{code}\n{export_block}\n}})();'


def main():
    with open(SRC, 'r', encoding='utf-8') as f:
        code = f.read()

    code = strip_comments(code)
    code = apply_renames(code)
    code = minify(code)
    code = create_wrapper(code)

    with open(DST, 'w', encoding='utf-8') as f:
        f.write(code)

    src_size = os.path.getsize(SRC)
    dst_size = os.path.getsize(DST)
    print(f'✓ {SRC} ({src_size:,} bytes) → {DST} ({dst_size:,} bytes)')
    print(f'  压缩率: {dst_size/src_size*100:.1f}%')


if __name__ == '__main__':
    main()
