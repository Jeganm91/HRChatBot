#!/bin/bash
set -e

# tr -d '\r' guards against the caller passing arguments with a stray trailing
# carriage return (seen from the trial platform's invocation) independent of
# this file's own line endings.
user_email="$(printf '%s' "$1" | tr -d '\r')"
sub="$(printf '%s' "$2" | tr -d '\r')"
location="southindia"
rgname=$(echo "$user_email" | cut -d "@" -f1)
git_repo="https://github.com/Jeganm91/HRChatBot.git"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# 1. Resource Group + identity lookups (fast, must complete first)
# ---------------------------------------------------------------------------
az group create -l "$location" -n "${rgname}" --subscription "$sub" > /dev/null
id=$(az ad user show --id "$user_email" --query "id" --output tsv)
subid=$(az account show --name "$sub" --query id -o tsv)
scope="/subscriptions/$subid/resourceGroups/$rgname"

rand=$(date +%s | tail -c 6)
oiname="hroi${rand}"

# ---------------------------------------------------------------------------
# 2. Role assignments + policy setup, fully in the background -- this app
# only needs Cognitive Services (Azure OpenAI). No Storage, no AI Search, no
# MCP server, so none of those roles/resources are created at all.
# MSYS_NO_PATHCONV stops Git-Bash on Windows from mangling the leading
# "/subscriptions/..." into a Windows path (seen firsthand: it silently
# turns into "C:/Program Files/Git/subscriptions/..." and every role
# assignment call fails with a confusing "MissingSubscription" error).
# ---------------------------------------------------------------------------
MSYS_NO_PATHCONV=1 az role assignment create --assignee "$id" --role "DenyPolicyDelete" --scope "$scope" > /dev/null 2>&1 &
MSYS_NO_PATHCONV=1 az role assignment create --assignee "$id" --role "Cognitive Services OpenAI Contributor" --scope "$scope" > /dev/null 2>&1 &

(
  polintid=$(az policy set-definition list --subscription "$sub" --query "[?displayName=='AISearch-Str-acr-containerapp-Int-Policy'].id" -o tsv)
  az policy assignment create --name "ContainerACR-$rgname" --display-name "ContainerACR-$rgname" --scope "$scope" --policy-set-definition "$polintid" --description "You can be able to create and configure AI Search, Open AI, VM, ACR, Storage in this exercise" > /dev/null 2>&1
  polname=$(az policy assignment list -g "$rgname" --subscription "$sub" --query [].name -o tsv)

  az policy assignment non-compliance-message create -n "$polname" -m "You are allowed to create and configure AI Search, Open AI, VM, ACR, Storage in this exercise" --policy-definition-reference-id Allow-Cognitive-Str-ACR-ContApp-Only_1 --scope "$scope" > /dev/null 2>&1 &
  az policy assignment non-compliance-message create -n "$polname" -m "You are allowed to create and configure GPT-5-mini in this exercise" --policy-definition-reference-id AIF-Gpt40mini-Txtemb-Policy_1 --scope "$scope" > /dev/null 2>&1 &
  az policy assignment non-compliance-message create -n "$polname" -m "You are allowed to create and configure 10K context limit for LLM and embedding model in this exercise" --policy-definition-reference-id Restrict-LLM-1L-Token-Policy_1 --scope "$scope" > /dev/null 2>&1 &
  az policy assignment non-compliance-message create -n "$polname" -m "You are allowed to create this service in central india, south india, west india regions" --policy-definition-reference-id Restrict-Location-Policy_1 --scope "$scope" > /dev/null 2>&1 &
  wait
) &
POLICY_PID=$!

# ---------------------------------------------------------------------------
# 3. Azure OpenAI account creation. Nothing else time-consuming needs to run
# before the VM can be deployed (no Storage account, no AI Search service, no
# blob upload, no indexer/skillset -- the knowledge base ships inside the git
# repo itself), so this is on the critical path rather than backgrounded:
# cloud-init needs the real endpoint/key baked in before the VM boots, not
# patched in afterward (which would race cloud-init's own env-file write).
# ---------------------------------------------------------------------------
echo "Creating OpenAI account ${oiname}..."
az cognitiveservices account create -n "$oiname" -g "$rgname" -l "$location" --kind OpenAI --sku S0 --subscription "$sub" > /dev/null

echo "Creating gpt-5-mini deployment..."
az cognitiveservices account deployment show --name "$oiname" -g "$rgname" --subscription "$sub" \
    --deployment-name "gpt-5-mini" > /dev/null 2>&1 || \
az cognitiveservices account deployment create --name "$oiname" -g "$rgname" --subscription "$sub" \
    --deployment-name "gpt-5-mini" --model-name "gpt-5-mini" \
    --model-version "2025-08-07" --model-format OpenAI --sku-capacity 10 --sku-name "GlobalStandard" > /dev/null

echo "Creating text-embedding-3-small deployment (semantic search, no Azure AI Search needed)..."
az cognitiveservices account deployment show --name "$oiname" -g "$rgname" --subscription "$sub" \
    --deployment-name "text-embedding-3-small" > /dev/null 2>&1 || \
az cognitiveservices account deployment create --name "$oiname" -g "$rgname" --subscription "$sub" \
    --deployment-name "text-embedding-3-small" --model-name "text-embedding-3-small" \
    --model-version "1" --model-format OpenAI --sku-capacity 10 --sku-name "GlobalStandard" > /dev/null

oi_endpoint=$(az cognitiveservices account show -n "$oiname" -g "$rgname" --subscription "$sub" --query "properties.endpoint" -o tsv)
oi_key=$(az cognitiveservices account keys list -n "$oiname" -g "$rgname" --subscription "$sub" --query "key1" -o tsv)

# ---------------------------------------------------------------------------
# 4. Build Cloud-Init Data for VM, with the real OpenAI endpoint/key already
# baked in. The app has no external knowledge-base store to provision --
# knowledge_base_docs/*.md ships inside the git repo itself and is read
# straight off local disk by app.py.
# ---------------------------------------------------------------------------
cloud_init_content=$(cat <<EOF
#cloud-config
package_upgrade: false
packages: [python3-pip, python3-venv, git, curl]
runcmd:
  - git clone ${git_repo} /opt/hr-lab
  - python3 -m venv /opt/hr-lab/.venv
  - /opt/hr-lab/.venv/bin/pip install --upgrade pip
  - /opt/hr-lab/.venv/bin/pip install -r /opt/hr-lab/requirements.txt
  - mkdir -p /etc/environment.d
  - |
    cat <<EOC > /etc/environment.d/hr-lab.conf
    AZURE_OPENAI_ENDPOINT=${oi_endpoint}
    AZURE_OPENAI_API_KEY=${oi_key}
    AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-5-mini
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small
    AZURE_OPENAI_API_VERSION=2024-02-15-preview
    EOC
  - |
    cat <<EOC > /etc/systemd/system/hr-app.service
    [Unit]
    Description=HR Chatbot Context Engineering Lab App
    After=network.target

    [Service]
    WorkingDirectory=/opt/hr-lab
    EnvironmentFile=/etc/environment.d/hr-lab.conf
    ExecStart=/opt/hr-lab/.venv/bin/python /opt/hr-lab/app.py
    Restart=always
    User=root

    [Install]
    WantedBy=multi-user.target
    EOC
  - systemctl daemon-reload
  - systemctl enable --now hr-app.service
EOF
)

custom_data_b64=$(echo "$cloud_init_content" | base64 | tr -d '\n')

# ---------------------------------------------------------------------------
# 5. Deploy VM and networking.
# ---------------------------------------------------------------------------
echo "Deploying VM..."
deploy_output=$(az deployment group create --name "infra-deploy-${rand}" --resource-group "$rgname" --subscription "$sub" --template-file "${SCRIPT_DIR}/deploy_infra.json" --parameters customDataBase64="$custom_data_b64" --query "properties.outputs" -o json)
# python3 instead of jq -- jq isn't guaranteed to be installed wherever this
# script is run from (e.g. a local Git-Bash shell), unlike Cloud Shell.
vm_principal_id=$(echo "$deploy_output" | python3 -c "import json,sys; print(json.load(sys.stdin)['vmPrincipalId']['value'])")
public_ip=$(echo "$deploy_output" | python3 -c "import json,sys; print(json.load(sys.stdin)['publicIpAddress']['value'])")

# VM managed-identity role -- only needs to reach Azure OpenAI, fires in the
# background, nothing waits on it. MSYS_NO_PATHCONV stops Git-Bash on Windows
# from mangling the leading "/subscriptions/..." into a Windows path.
MSYS_NO_PATHCONV=1 az role assignment create --assignee "$vm_principal_id" --role "Cognitive Services OpenAI Contributor" --scope "$scope" > /dev/null 2>&1 &

# Wait for policy setup and any remaining background role assignments
wait $POLICY_PID
wait

echo "Success"
echo "App URL: http://${public_ip}:8080"
