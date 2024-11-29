import pkg_resources

# 讀取 requirements.txt 文件
with open("requirements.txt", "r") as file:
    requirements = file.readlines()

# 檢查每個包是否已安裝且版本匹配
for requirement in requirements:
    requirement = requirement.strip()  # 去除空白
    if not requirement or requirement.startswith("#"):
        continue  # 忽略空行和註解

    try:
        # 解析 requirement，例如：'numpy==1.21.0'
        pkg_name, pkg_version = requirement.split("==")
        installed_version = pkg_resources.get_distribution(pkg_name).version

        # 比較版本
        if installed_version == pkg_version:
            print(f"{pkg_name}=={pkg_version} 已正確安裝")
        else:
            print(f"{pkg_name} 已安裝，但版本為 {installed_version}，預期為 {pkg_version}")

    except pkg_resources.DistributionNotFound:
        print(f"{pkg_name} 未安裝")

    except ValueError:
        print(f"無法解析 requirement：{requirement}")
