# 라이브러리들을 불러옵니다
import os
import csv
import numpy as np  # 수치 계산을 위한 라이브러리
from pathlib import Path  # 파일 경로를 다루기 위한 라이브러리

# pymatgen의 핵심 기능들을 불러옵니다
from pymatgen.core import Structure  # 결정 구조를 다루는 클래스
from pymatgen.io.vasp.sets import MPStaticSet  # VASP 입력 파일 세트 클래스
from pymatgen.io.vasp.inputs import Kpoints  # k-points 설정을 위한 클래스
from mp_api.client import MPRester as MPClient  # 최신 버전의 API 클라이언트

# 🔑 여기에 당신의 API 키를 직접 입력하세요!
api_key = os.getenv("MP_API_KEY") or os.getenv("PMG_MAPI_KEY")
if not api_key:
    raise RuntimeError("Set MP_API_KEY (or PMG_MAPI_KEY) in your environment.")


# CSV 파일에서 MP ID들을 읽어오는 함수
def read_mp_ids_from_csv(csv_path):
    """
    CSV 파일에서 Materials Project ID들을 읽어오는 함수
    
    Args:
        csv_path: CSV 파일 경로
        
    Returns:
        MP ID들의 리스트
    """
    mp_ids = []
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)  # 첫 번째 줄을 헤더로 사용
            
            for row in reader:
                if 'material_id' in row and row['material_id'].strip():
                    mp_ids.append(row['material_id'].strip())
                    
        print(f"✅ CSV 파일에서 {len(mp_ids)}개의 MP ID를 성공적으로 읽어왔습니다!")
        print(f"첫 5개 ID: {mp_ids[:5]}")
        
    except FileNotFoundError:
        print(f"❌ 파일을 찾을 수 없습니다: {csv_path}")
    except Exception as e:
        print(f"❌ CSV 파일 읽기 중 오류 발생: {e}")
    
    return mp_ids

# CSV 파일 경로 설정
csv_file_path = "/Users/jaekwansmac/Desktop/MP_dielectric_constant/perovskite_dielectric_bec_data.csv"

# MP ID들 읽어오기
mp_ids = read_mp_ids_from_csv(csv_file_path)

# ===== 구조 불러오기 함수 =====
def get_structure_from_mp(mp_id, api_key):
    """
    Materials Project에서 구조를 가져오는 함수
    
    Args:
        mp_id: Materials Project ID
        api_key: API 키
        
    Returns:
        Structure 객체
    """
    with MPClient(api_key) as mpr:
        structure = mpr.get_structure_by_material_id(mp_id)
    return structure



# ===== 공통 INCAR 설정 =====
# - MPStaticSet을 기반으로 하되, PEAD finite-field + LCALCEPS + 수렴강화
# - Γ-centered 정규 메쉬 필요(PEAD는 Γ 포함 정규 메쉬 요구)
#   (MPStaticSet은 기본적으로 MP 규칙을 따르며, KPOINTS는 자동 생성됨)

common_incar = {
    "LCALCEPS": False,    # 유전상수 계산 비활성화 (전기장 계산에서는 불필요)
    "LCALCPOL": True,     # 분극 계산 활성화 (전기장 효과 계산에 필요)
    "ISMEAR": 0,          # Gaussian smearing 사용 (금속이 아닌 경우)
    "SIGMA": 0.05,        # smearing 너비 (eV 단위)
    "EDIFF": 1e-6,        # 전자 수렴 기준 (에너지 차이)
    "PREC": "Accurate",   # 정밀도 설정 (고정밀 계산)
    "ADDGRID": True,      # 추가 그리드 포인트 사용 (정확도 향상)
    "LREAL": False,       # 실공간 계산 비활성화 (k-공간 계산 사용)
    "LCHARG": False,      # CHGCAR 파일 저장 비활성화 (디스크 공간 절약)
    "LWAVE": False,       # WAVECAR 파일 저장 비활성화 (디스크 공간 절약)
    "ENCUT": 520,         # 플래넬 파동 함수 컷오프 에너지 (eV)
    "NSW": 0,             # 이온 스텝 수 (0 = 이온 위치 고정)
    "IBRION": -1,         # 이온 최적화 방법 (-1 = 이온 위치 고정)
    "ISYM": 0,            # 대칭성 사용 비활성화 (전기장이 대칭성을 깨뜨림)
    "SYMPREC": 1e-8,      # 대칭성 인식 정밀도 (ISYM=0이므로 사용되지 않음)
    "LASPH": True,        # 비구면 기여 활성화 (정확한 계산을 위해 필요)
}


print("✅ INCAR 설정이 완료되었습니다!")
print("주요 설정:")
print(f"  - LCALCEPS: {common_incar['LCALCEPS']} (유전상수 계산)")
print(f"  - ISMEAR: {common_incar['ISMEAR']} (Gaussian smearing)")
print(f"  - EDIFF: {common_incar['EDIFF']} (수렴 기준)")
print(f"  - PREC: {common_incar['PREC']} (정확도)")


# 모든 CSV 물질에 대해 필드 스윕 입력을 생성하는 함수
def create_field_sweep_for_all_materials(mp_ids, max_materials=3, E_mags=[0.001, 0.003, 0.005]):
    """
    CSV에서 읽어온 모든 MP ID에 대해 필드 스윕 입력을 생성하는 함수
    
    Args:
        mp_ids: MP ID들의 리스트
        max_materials: 최대 처리할 물질 수
        E_mags: 전기장 세기 리스트
    """
    if not mp_ids:
        print("❌ 처리할 MP ID가 없습니다!")
        return
    
    # 처리할 물질 수 제한
    materials_to_process = mp_ids[:max_materials]
    print(f"🔧 {len(materials_to_process)}개 물질에 대해 필드 스윕 입력을 생성합니다...")
    
    success_count = 0
    fail_count = 0
    
    for i, mp_id in enumerate(materials_to_process, 1):
        print(f"\\n{'='*60}")
        print(f"📋 {i}/{len(materials_to_process)}: {mp_id} 처리 중")
        print(f"{'='*60}")
        
        try:
            # 구조 가져오기
            structure = get_structure_from_mp(mp_id, api_key)
            print(f"✅ 구조 로드 완료: {structure.composition.reduced_formula}")
            
            # 루트 디렉토리 생성
            root = Path(f"finite_field_sweep_{mp_id}")
            root.mkdir(exist_ok=True)
            
            # k-points 설정 (Γ-centered 정규 메쉬)
            kpoints = Kpoints.gamma_automatic(kpts=(8,8,8))
            print(f"✅ k-points 설정: Γ-centered 8x8x8 메쉬")
            
            # Zero-field 참고용 입력 생성
            zf_dir = root / "E0_ref"
            zf_dir.mkdir(exist_ok=True)
            zf_incar = {**common_incar}             # <-- EFIELD_PEAD 키 제거!
            vset_zf = MPStaticSet(
                structure,
                user_incar_settings=zf_incar,
                user_kpoints_settings=kpoints
            )
            vset_zf.write_input(str(zf_dir), potcar_spec=True)

            print(f"✅ Zero-field 참고 입력 생성: {zf_dir.name}")
            
            # ===== c축 방향 스윕만 생성 =====
            # c축 단위벡터 계산 (결정학적 [001] 방향)
            c_hat = structure.lattice.matrix[2] / np.linalg.norm(structure.lattice.matrix[2])
            print(f"✅ c축 단위벡터: [{c_hat[0]:.4f}, {c_hat[1]:.4f}, {c_hat[2]:.4f}]")
            
            total_dirs = 0
            for E in E_mags:
                # c축 방향으로 전기장 벡터 계산: E * c_hat
                evec = (c_hat * E).tolist()
                tag = f"E_{E:.4f}_along_001"
                wdir = root / tag
                wdir.mkdir(parents=True, exist_ok=True)
                
                field_incar = {**common_incar, "EFIELD_PEAD": evec,"SKIP_EDOTP": True}
                vset = MPStaticSet(structure, user_incar_settings=field_incar, user_kpoints_settings=kpoints)
                vset.write_input(str(wdir), potcar_spec=True)
                
                total_dirs += 1
                print(f"✅ {tag} 생성 완료 (c축 방향)")
            
            print(f"✅ {mp_id} 필드 스윕 입력 생성 완료! ({total_dirs + 1}개 디렉토리)")
            print(f"   - Zero-field: 1개")
            print(f"   - c축 방향: {len(E_mags)}개")
            success_count += 1
                
        except Exception as e:
            print(f"❌ {mp_id} 처리 중 오류: {e}")
            fail_count += 1
    
    print(f"\\n🎉 전체 처리 완료!")
    print(f"✅ 성공: {success_count}개")
    print(f"❌ 실패: {fail_count}개")
    print(f"📊 성공률: {success_count/(success_count+fail_count)*100:.1f}%")

# 사용자 설정
MAX_MATERIALS = 256  # 테스트용으로 3개만 처리 (필요시 더 늘릴 수 있음)
E_FIELD_MAGNITUDES = [-0.003, -0.001, 0.001, 0.003]  # 전기장 세기 (eV/Å)

print(f"⚙️  설정:")
print(f"   - 최대 처리 물질 수: {MAX_MATERIALS}개")
print(f"   - 전기장 세기: {E_FIELD_MAGNITUDES} eV/Å")
print(f"   - 전기장 방향: c축 방향 (결정학적 [001] 방향)")

# 모든 물질에 대해 필드 스윕 입력 생성
if mp_ids:
    create_field_sweep_for_all_materials(
        mp_ids=mp_ids,
        max_materials=MAX_MATERIALS,
        E_mags=E_FIELD_MAGNITUDES
    )
else:
    print("⚠️  먼저 위의 셀을 실행해서 MP ID들을 읽어와주세요!")