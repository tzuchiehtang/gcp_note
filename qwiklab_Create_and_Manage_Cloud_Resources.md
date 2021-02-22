
# GCP Notes

- [GCP Notes](#gcp-notes)
  - [First Step](#first-step)
  - [To see what your default region and zone settings are](#to-see-what-your-default-region-and-zone-settings-are)
  - [Set the default region and zone for all resources](#set-the-default-region-and-zone-for-all-resources)
  - [Creating a Virtual Machine Instance with `gcloud`](#creating-a-virtual-machine-instance-with-gcloud)
  - [Create a GKE cluster](#create-a-gke-cluster)
  - [Set Up Network and HTTP Load Balancers](#set-up-network-and-http-load-balancers)

---

## First Step

1. list the active account name and get permission:

    ```bash
    gcloud auth list
    ```

2. list the project ID:

    ```bash
    gcloud config list project
    ```

3. Regions and Zones:
    > A region is a specific geographical location where you can run your resources. Each region has one or more zones.
    - For example:
        | Region        | Zones         |
        | ------------- | ------------- |
        |  Central US   | us-central1-a |
        |               | us-central1-b |
        |               | us-central1-c |

    1. Identify your default region and zone:
        - Copy your project ID to your clipboard or text editor. The project ID is listed in 2 places:
          1. In the Google Cloud Console, on the Dashboard, under Project info. (Click Navigation menu (Navigation menu), and then click Home > Dashboard.)

          2. On the Qwiklabs tab near your username and password.

        ```bash
        gcloud compute project-info describe --project <your_project_ID>
        ```

## To see what your default region and zone settings are

```bash
gcloud config get-value compute/zone
gcloud config get-value compute/region
```

## Set the default region and zone for all resources

```bash
gcloud config set compute/zone [ZONE](us-central1-a)
gcloud config set compute/region [REGION](us-central1)
```

## Creating a Virtual Machine Instance with `gcloud`

1. Create command:

    ```bash
    gcloud compute instances create [INSTANCE-NAME] --machine-type n1-standard-2 --zone us-central1-c
    ```

2. parameters:

    | Field | Value | Additional Information|
    | ------------- | ------------- | ------------- |
    | Name | gcelab | Name for the VM instance|
    | Region | us-central1 (Iowa) | For more information about regions, see Regions and Zones.|
    | Zone | us-central1-c | Note: Remember the zone that you selected: you'll need it later. For more information about zones, see Regions and Zones.|
    | Series | N1 | Name of the series|
    | Machine Type | 2 vCPUs | This is an (n1-standard-2), 2-CPU, 7.5GB RAM instance. Several machine types are available, ranging from micro instance types to 32-core/208GB RAM instance types. For more information, see Machine Types. Note: A new project has a default resource quota, which may limit the number of CPU cores. You can request more when you work on projects outside this lab.|
    | Boot Disk | New 10 GB standard persistent disk OS Image: Debian GNU/Linux 10 (buster) | Several images are available, including Debian, Ubuntu, CoreOS, and premium images such as Red Hat Enterprise Linux and Windows Server. For more information, see Operating System documentation.|
    | Firewall | Allow HTTP traffic | Select this option in order to access a web server that you'll install later. Note: This will automatically create a firewall rule to allow HTTP traffic on port 80.|

3. Connect to instance with SSH:
    > Make sure to add your zone, or omit the --zone flag if you've set the option globally:

    ```bash
    gcloud compute ssh INSTANCE_NAME --zone us-central1-c
    ```

## Create a GKE cluster

1. create a cluster:

    ```bash
    gcloud container clusters create [CLUSTER-NAME]
    ```

2. Get authentication credentials for the cluster:

    ```bash
    gcloud container clusters get-credentials [CLUSTER-NAME]
    ```

3. Deploy an application to the cluster:
    1. To create a new Deployment hello-server from the hello-app container image, run the following kubectl create command

        ```bash
        kubectl create deployment [APPLICATION_NAME](hello-server) --image=gcr.io/google-samples/hello-app:1.0
        ```

        >In this case, --image specifies a container image to deploy. The command pulls the example image from a Container Registry bucket. gcr.io/google-samples/hello-app:1.0 indicates the specific image version to pull. If a version is not specified, the latest version is used.

    2. To create a Kubernetes Service, which is a Kubernetes resource that lets you expose your application to external traffic:

        ```bash
        kubectl expose deployment [APPLICATION_NAME](hello-server) --type=LoadBalancer --port [PORT](8080)
        ```

        In this command:
        1. --port specifies the port that the container exposes.
        2. type="LoadBalancer" creates a Compute Engine load balancer for your container.

    3. To inspect the hello-server Service:

        ```bash
        kubectl get service
        ```

        Expected output :

        ```bash
        NAME              TYPE              CLUSTER-IP        EXTERNAL-IP      PORT(S)           AGE
        hello-server      loadBalancer      10.39.244.36      35.202.234.26    8080:31991/TCP    65s
        kubernetes        ClusterIP         10.39.240.1       <none>           433/TCP           5m13s
        ```

4. Deleting the cluster:

    ```bash
    gcloud container clusters delete [CLUSTER-NAME]
    ```

## Set Up Network and HTTP Load Balancers

1. Create multiple web server instances

    ```bash
    gcloud compute instances create www1 \
        --image-family debian-9 \
        --image-project debian-cloud \
        --zone us-central1-a \
        --tags network-lb-tag \
        --metadata startup-script="#! /bin/bash
            sudo apt-get update
            sudo apt-get install apache2 -y
            sudo service apache2 restart
            echo '<!doctype html><html><body><h1>www1</h1></body></html>' | tee /var/www/html/index.html"
    ```

    ```bash
    gcloud compute instances create www2 \
        --image-family debian-9 \
        --image-project debian-cloud \
        --zone us-central1-a \
        --tags network-lb-tag \
        --metadata startup-script="#! /bin/bash
            sudo apt-get update
            sudo apt-get install apache2 -y
            sudo service apache2 restart
            echo '<!doctype html><html><body><h1>www2</h1></body></html>' | tee /var/www/html/index.html"
    ```

    ```bash
    gcloud compute instances create www3 \
        --image-family debian-9 \
        --image-project debian-cloud \
        --zone us-central1-a \
        --tags network-lb-tag \
        --metadata startup-script="#! /bin/bash
            sudo apt-get update
            sudo apt-get install apache2 -y
            sudo service apache2 restart
            echo '<!doctype html><html><body><h1>www3</h1></body></html>' | tee /var/www/html/index.html"
    ```

2. Create a firewall rule to allow external traffic to the VM instances:

     ```bash
     gcloud compute firewall-rules create www-firewall-network-lb \
         --target-tags network-lb-tag --allow tcp:80
     ```

    1. Run the following to list your instances. You'll see their IP addresses in the **`EXTERNAL_IP`** column:

        ```bash
        gcloud compute instances list
        ```

3. Configure the [load balancing service](https://cloud.google.com/load-balancing/docs/network)

    1. Create a static external IP address for your load balancer:

        ```bash
        gcloud compute addresses create network-lb-ip-1 \
            --region us-central1
        ```

    2. Add a legacy HTTP health check resource:

        ```bash
        gcloud compute http-health-checks create basic-check
        ```

    3. Add a target pool and use the health check in the same region as your instances.

        ```bash
        gcloud compute target-pools create www-pool \
            --region us-central1 --http-health-check basic-check
        ```

    4. Add the instances to the pool:

        ```bash
        gcloud compute target-pools add-instances www-pool \
            --instances www1,www2,www3
        ```

    5. Add a forwarding rule:

        ```bash
        gcloud compute forwarding-rules create www-rule \
            --region us-central1 \
            --ports 80 \
            --address network-lb-ip-1 \
            --target-pool www-pool
        ```

3. Sending traffic to your instances:

    Enter the following command to view the external IP address of the www-rule forwarding rule used by the load balancer:

    ```bash
    gcloud compute forwarding-rules describe www-rule --region us-central1
    ```

    Use curl command to access the external IP address, replacing IP_ADDRESS with an external IP address from the previous command:

    ```bash
    while true; do curl -m1 IP_ADDRESS; done
    ```

4. Create an HTTP load balancer:

    HTTP(S) Load Balancing is implemented on Google Front End (GFE). GFEs are distributed globally and operate together using Google's global network and control plane.

    Requests are always routed to the instance group that is closest to the user, if that group has enough capacity and is appropriate for the request. If the closest group does not have enough capacity, the request is sent to the closest group that does have capacity.

    1. Create the **load balancer template**.

        ```bash
        gcloud compute instance-templates create lb-backend-template \
            --region=us-central1 \
            --network=default \
            --subnet=default \
            --tags=allow-health-check \
            --image-family=debian-9 \
            --image-project=debian-cloud \
            --metadata=startup-script='#! /bin/bash
                apt-get update
                apt-get install apache2 -y
                a2ensite default-ssl
                a2enmod ssl
                vm_hostname="$(curl -H "Metadata-Flavor:Google" \
                http://169.254.169.254/computeMetadata/v1/instance/name)"
                echo "Page served from: $vm_hostname" | \
                tee /var/www/html/index.html
                systemctl restart apache2'
        ```

    2. Create a **managed instance group** based on the template.

        ```bash
        gcloud compute instance-groups managed create lb-backend-group \
            --template=lb-backend-template --size=2 --zone=us-central1-a
        ```

    3. Create the `fw-allow-health-check` **firewall rule**.

        This is an ingress rule that allows traffic from the Google Cloud health checking systems (`130.211.0.0/22` and `35.191.0.0/16`). This lab uses the target tag `allow-health-check` to identify the VMs.

        ```bash
        gcloud compute firewall-rules create fw-allow-health-check \
            --network=default \
            --action=allow \
            --direction=ingress \
            --source-ranges=130.211.0.0/22,35.191.0.0/16 \
            --target-tags=allow-health-check \
            --rules=tcp:80
        ```

    4. Set up a **global static external IP address** that using to reach your load balancer.

        ```bash
        gcloud compute addresses create lb-ipv4-1 \
            --ip-version=IPV4 \
            --global
        ```

        > Note the IPv4 address that was reserved:

        ```bash
        gcloud compute addresses describe lb-ipv4-1 \
            --format="get(address)" \
            --global
        ```

    5. Create a **healthcheck** for the load balancer.

        ```bash
        gcloud compute health-checks create http http-basic-check \
            --port 80
        ```

    6. Create a **backend service**.

        ```bash
        gcloud compute backend-services create web-backend-service \
            --protocol=HTTP \
            --port-name=http \
            --health-checks=http-basic-check \
            --global
        ```

    7. Add instance group as the backend to the backend service.

        ```bash
        gcloud compute backend-services add-backend web-backend-service \
            --instance-group=lb-backend-group \
            --instance-group-zone=us-central1-a \
            --global
        ```

    8. Create a **URL map** to route the incoming requests to the default backend service.

        ```bash
        gcloud compute url-maps create web-map-http \
            --default-service web-backend-service
        ```

    9. Create a **target HTTP proxy** to route requests to your URL map.

        ```bash
        gcloud compute target-http-proxies create http-lb-proxy \
            --url-map web-map-http
        ```

    10. Create a **global forwarding rule** to route incoming requests to the proxy:

        ```bash
        gcloud compute forwarding-rules create http-content-rule \
            --address=lb-ipv4-1\
            --global \
            --target-http-proxy=http-lb-proxy \
            --ports=80
        ```
